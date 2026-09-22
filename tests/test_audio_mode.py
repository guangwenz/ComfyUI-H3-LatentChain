"""Regression test for MiniMaxH3TakeGuide audio_mode ("carry" vs "fresh").

Runs guide() with stubbed ComfyUI modules (no ComfyUI import needed) and
asserts the SPEC_takeguide_audio_mode.md contract:

1. "carry" pins the previous take's audio window into the new latent head
   and zeroes the audio mask there (legacy behavior, unchanged).
2. "fresh" leaves new_audio / audio_mask untouched (video-only handoff):
   the audio head stays the take's own wav-conditioned latent.
3. Both modes pin the identical video edge tokens, return the identical
   keyframe anchor, and agree on `covered`.

Stub values match the installed core exactly:
- comfy.ldm.minimax.model.FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
- context=22 -> frames=22, prefix_tokens=7, audio_window=37
- covered = _frame_count(7) = 22 (the spec's number)
- guide mode pins only the last GUIDE_HANDOFF_VIDEO_TOKENS=2 edge tokens.
"""

from __future__ import annotations

import sys
import types
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- stub ComfyUI modules before importing chain -----------------------------
nested_tensor_mod = types.ModuleType("comfy.nested_tensor")


class NestedTensor:
    def __init__(self, tensors):
        self.tensors = tuple(tensors)


nested_tensor_mod.NestedTensor = NestedTensor
comfy_mod = types.ModuleType("comfy")
comfy_mod.nested_tensor = nested_tensor_mod
sys.modules["comfy"] = comfy_mod
sys.modules["comfy.nested_tensor"] = nested_tensor_mod

model_mod = types.ModuleType("comfy.ldm.minimax.model")
model_mod.FRAME_PER_TOKEN = (1, 4, 4, 4, 4)  # real core value
sys.modules["comfy.ldm"] = types.ModuleType("comfy.ldm")
sys.modules["comfy.ldm.minimax"] = types.ModuleType("comfy.ldm.minimax")
sys.modules["comfy.ldm.minimax.model"] = model_mod

folder_paths_mod = types.ModuleType("folder_paths")
folder_paths_mod.get_input_directory = lambda: os.path.join(os.path.dirname(__file__), "_tmp_in")
folder_paths_mod.get_output_directory = lambda: os.path.join(os.path.dirname(__file__), "_tmp_out")
sys.modules["folder_paths"] = folder_paths_mod

node_helpers_mod = types.ModuleType("node_helpers")


def conditioning_set_values(positive, values):
    base = positive[0][1]
    merged = {**base, **values}
    return [[positive[0][0], merged]]


node_helpers_mod.conditioning_set_values = conditioning_set_values
sys.modules["node_helpers"] = node_helpers_mod

import torch  # noqa: E402

import chain  # noqa: E402


def main():
    torch.manual_seed(0)
    B, T_ref, T_new, Hv, Wv, A = 1, 20, 15, 4, 4, 60
    ref_video = torch.randn(B, 24, T_ref, Hv, Wv)
    ref_audio = torch.randn(B, 32, 2, A)
    take = chain.H3Take(video=ref_video, audio=ref_audio, meta={"frames": 22})

    new_video = torch.randn(B, 24, T_new, Hv, Wv)
    new_audio = torch.randn(B, 32, 2, A)
    positive = [[torch.zeros(1), {}]]
    latent = {"samples": (new_video, new_audio)}

    frames, prefix_tokens, audio_window = chain._temporal_shape_or_local(22)
    assert (frames, prefix_tokens, audio_window) == (22, 7, 37), \
        f"stub grid drifted: {(frames, prefix_tokens, audio_window)}"
    edge = chain.GUIDE_HANDOFF_VIDEO_TOKENS  # 2

    cond_c, lat_c, cov_c = chain.MiniMaxH3TakeGuide.guide(
        positive, latent, 22, take=take, frame_idx=0, mode="guide", audio_mode="carry"
    )
    cond_f, lat_f, cov_f = chain.MiniMaxH3TakeGuide.guide(
        positive, latent, 22, take=take, frame_idx=0, mode="guide", audio_mode="fresh"
    )

    # --- shared invariants (both modes) ---
    expected_covered = chain._frame_count(prefix_tokens)
    assert cov_c == cov_f == expected_covered == 22, \
        f"covered frames changed: {cov_c} vs {cov_f} (expected 22)"
    kf_c = cond_c[0][1]["minimax_keyframes"]
    kf_f = cond_f[0][1]["minimax_keyframes"]
    assert kf_c[0]["resolved_frame_index"] == kf_f[0]["resolved_frame_index"] == 0
    assert torch.equal(kf_c[0]["latent"], kf_f[0]["latent"]), \
        "keyframe anchor differs between modes"
    window = chain._tail_tokens(ref_video, prefix_tokens)

    video_c, audio_c = lat_c["samples"].tensors
    video_f, audio_f = lat_f["samples"].tensors
    vmask_c, amask_c = lat_c["noise_mask"].tensors
    vmask_f, amask_f = lat_f["noise_mask"].tensors

    # video branch identical in both modes: head tokens [prefix-edge:prefix] pinned + frozen
    assert torch.equal(video_c, video_f), "video handoff differs between modes"
    assert torch.equal(vmask_c, vmask_f), "video mask differs between modes"
    assert torch.equal(video_f[:, :, prefix_tokens - edge:prefix_tokens],
                       window[:, :, prefix_tokens - edge:prefix_tokens]), \
        "video edge tokens not pinned to ref tail"
    assert float(vmask_f[:, :, prefix_tokens - edge:prefix_tokens].min()) == 0.0, \
        "video edge not frozen"
    assert float(vmask_f[:, :, :prefix_tokens - edge].min()) == 1.0, \
        "video head body unexpectedly frozen"
    assert float(vmask_f[:, :, prefix_tokens:].min()) == 1.0, \
        "video tail unexpectedly frozen"

    # carry: audio head == ref audio's trailing window, frozen there (legacy)
    carry_window = chain._fit_window(
        chain._context_audio_window(ref_audio, audio_window), audio_window
    )
    assert torch.equal(audio_c[..., :audio_window], carry_window), "carry audio head wrong"
    assert float(amask_c[..., :audio_window].min()) == 0.0, "carry audio head not frozen"
    assert float(amask_c[..., audio_window:].min()) == 1.0, "carry audio body unexpectedly frozen"

    # fresh: audio plane fully untouched (values AND mask), no ref-audio leak
    assert torch.equal(audio_f, new_audio), "fresh mutated the take's own audio"
    assert float(amask_f.min()) == 1.0, "fresh froze part of the audio mask"
    assert not torch.equal(audio_f[..., :audio_window], carry_window), "fresh leaked ref audio"

    print("OK: carry/fresh behave per spec; covered=22; video handoff identical in both modes")


if __name__ == "__main__":
    main()
