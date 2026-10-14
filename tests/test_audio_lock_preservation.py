"""Regression test: MiniMaxH3TakeGuide preserves an incoming audio lock.

2026-10-14 root cause (s03 chained dialogue lip-drift): MiniMaxH3Easy
digital_human mode locks the driving track into the target audio latent with a
ZERO denoise mask (_lock_audio_latent). TakeGuide rebuilt the audio mask as
all-ones, discarding the lock, and (audio_mode="carry") overwrote the latent's
audio head with the previous take's audio. The sampler then fully re-denoised
the audio stream: H3 re-spoke the line on an invented schedule (measured
rendition onsets 1.3-6.5s late; head takes â€” which pass through TakeGuide
untouched â€” were always frame-exact).

Contract after the fix:
1. Incoming latent with audio mask containing any value < 1 (the lock):
   - audio stream passes through BYTE-IDENTICAL (no carry overwrite);
   - audio mask passes through IDENTICAL (still fully locked);
   - video seam pins/keyframes/covered unchanged;
   - audio_mode is ignored for the audio plane (carry and fresh agree).
2. Incoming latent WITHOUT an audio lock (no mask / all-ones mask): legacy
   carry/fresh behavior unchanged (head pin + mask, per test_audio_mode.py).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- stub ComfyUI modules before importing chain (same stubs as test_audio_mode) ---
nested_tensor_mod = types_module = __import__("types")
nested_tensor_mod = types_module.ModuleType("comfy.nested_tensor")


class NestedTensor:
    def __init__(self, tensors):
        self.tensors = tuple(tensors)

    def unbind(self):
        return self.tensors


nested_tensor_mod.NestedTensor = NestedTensor
comfy_mod = types_module.ModuleType("comfy")
comfy_mod.nested_tensor = nested_tensor_mod
sys.modules["comfy"] = comfy_mod
sys.modules["comfy.nested_tensor"] = nested_tensor_mod

model_mod = types_module.ModuleType("comfy.ldm.minimax.model")
model_mod.FRAME_PER_TOKEN = (1, 4, 4, 4, 4)  # real core value
sys.modules["comfy.ldm"] = types_module.ModuleType("comfy.ldm")
sys.modules["comfy.ldm.minimax"] = types_module.ModuleType("comfy.ldm.minimax")
sys.modules["comfy.ldm.minimax.model"] = model_mod

folder_paths_mod = types_module.ModuleType("folder_paths")
folder_paths_mod.get_input_directory = lambda: os.path.join(os.path.dirname(__file__), "_tmp_in")
folder_paths_mod.get_output_directory = lambda: os.path.join(os.path.dirname(__file__), "_tmp_out")
sys.modules["folder_paths"] = folder_paths_mod

node_helpers_mod = types_module.ModuleType("node_helpers")


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
    tts_audio = torch.randn(B, 32, 2, A)  # the digital-human driving track latent
    positive = [[torch.zeros(1), {}]]

    frames, prefix_tokens, audio_window = chain._temporal_shape_or_local(22)
    edge = chain.GUIDE_HANDOFF_VIDEO_TOKENS

    locked_video_mask = torch.ones(B, 1, T_new, Hv, Wv)
    locked_audio_mask = torch.zeros(B, 1, 2, A)  # fully locked driving track
    locked_latent = {
        "samples": chain.comfy.nested_tensor.NestedTensor((new_video, tts_audio)),
        "noise_mask": chain.comfy.nested_tensor.NestedTensor(
            (locked_video_mask, locked_audio_mask)
        ),
    }

    # 1) locked incoming + carry: audio plane must survive verbatim
    cond_l, lat_l, cov_l = chain.MiniMaxH3TakeGuide.guide(
        [[torch.zeros(1), {}]], locked_latent, 22, take=take, frame_idx=0,
        mode="guide", audio_mode="carry",
    )
    video_l, audio_l = lat_l["samples"].tensors
    vmask_l, amask_l = lat_l["noise_mask"].tensors
    assert torch.equal(audio_l, tts_audio), "carry overwrote the locked driving track"
    assert torch.equal(amask_l, locked_audio_mask), "audio lock mask was not preserved"
    assert float(amask_l.max()) == 0.0, "audio mask is no longer fully locked"
    # video seam still applies on a locked take
    window = chain._tail_tokens(ref_video, prefix_tokens)
    assert torch.equal(video_l[:, :, prefix_tokens - edge:prefix_tokens],
                       window[:, :, prefix_tokens - edge:prefix_tokens]), \
        "locked take lost the video seam pin"
    assert float(vmask_l[:, :, prefix_tokens - edge:prefix_tokens].min()) == 0.0
    assert cov_l == chain._frame_count(prefix_tokens)

    # 2) locked incoming + fresh: identical audio outcome (mode ignored when locked)
    cond_f2, lat_f2, _ = chain.MiniMaxH3TakeGuide.guide(
        [[torch.zeros(1), {}]], locked_latent, 22, take=take, frame_idx=0,
        mode="guide", audio_mode="fresh",
    )
    _, audio_f2 = lat_f2["samples"].tensors
    _, amask_f2 = lat_f2["noise_mask"].tensors
    assert torch.equal(audio_f2, tts_audio), "fresh mutated the locked driving track"
    assert torch.equal(amask_f2, locked_audio_mask), "fresh altered the audio lock mask"

    # 3) carry with an audio mask that is all-ones (not a lock): legacy behavior â€”
    #    the carry pin still applies (unlocked generated-audio continuity).
    ones_audio_mask = torch.ones(B, 1, 2, A)
    unlocked_latent = {
        "samples": chain.comfy.nested_tensor.NestedTensor((new_video, tts_audio)),
        "noise_mask": chain.comfy.nested_tensor.NestedTensor(
            (torch.ones(B, 1, T_new, Hv, Wv), ones_audio_mask)
        ),
    }
    cond_u, lat_u, _ = chain.MiniMaxH3TakeGuide.guide(
        [[torch.zeros(1), {}]], unlocked_latent, 22, take=take, frame_idx=0,
        mode="guide", audio_mode="carry",
    )
    _, audio_u = lat_u["samples"].tensors
    _, amask_u = lat_u["noise_mask"].tensors
    carry_window = chain._fit_window(
        chain._context_audio_window(ref_audio, audio_window), audio_window
    )
    assert torch.equal(audio_u[..., :audio_window], carry_window), \
        "legacy carry pin changed for an unlocked take"
    assert float(amask_u[..., :audio_window].min()) == 0.0
    assert float(amask_u[..., audio_window:].min()) == 1.0

    # 4) no mask at all: legacy behavior unchanged (original contract test covers
    #    this; here just verify carry still pins with a mask-less latent).
    plain_latent = {"samples": (new_video, tts_audio)}
    cond_p, lat_p, _ = chain.MiniMaxH3TakeGuide.guide(
        [[torch.zeros(1), {}]], plain_latent, 22, take=take, frame_idx=0,
        mode="guide", audio_mode="carry",
    )
    _, audio_p = lat_p["samples"].tensors
    assert torch.equal(audio_p[..., :audio_window], carry_window), \
        "legacy carry pin changed for a mask-less latent"

    print("OK: audio lock preserved through the chain seam; legacy carry unchanged")


if __name__ == "__main__":
    main()
