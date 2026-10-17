"""Regression test: MiniMaxH3AudioWindowLock rebuilds a dialogue audio lock
into a windowed one (speech verbatim, ambience generated elsewhere).

2026-10-16 stage-0 root cause: MiniMaxH3Easy digital_human locks the ENTIRE
audio timeline (audio_mask all zeros over the full grid), so dialogue takes
render speech + dead room tone while ambient (FL2VA) takes get full generated
soundscapes. The fix under test rebuilds the packed audio mask: 0 inside the
speech window (with margins), 1 outside, optional linear ramps at the edges.

Contracts:
1. The audio STREAM is rewritten: verbatim inside the locked core window,
   zeroed outside. Zeroing removes Easy's ``_fit_audio_latent`` last-frame
   repeat tail, which otherwise leaks quasi-speech content into the generated
   region and reconstructs the dialogue line (measured 2026-10-17).
2. The video stream and video mask are untouched.
3. Mask geometry: zeros exactly on [lo, hi), ramps outside the edges when
   ramp_steps > 0, ones elsewhere; lo/hi derived from seconds * steps-per-second
   on the clip's own audio grid.
4. Downstream TakeGuide compatibility: the windowed mask still counts as an
   audio lock (any < 1) and passes through the seam verbatim.
5. A latent without a noise_mask (unlocked take) is rejected loudly.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- stub ComfyUI modules before importing chain (same stubs as test_audio_mode) ---
types_module = __import__("types")
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
node_helpers_mod.conditioning_set_values = lambda positive, values: [
    [positive[0][0], {**positive[0][1], **values}]
]
sys.modules["node_helpers"] = node_helpers_mod

import torch  # noqa: E402

import chain  # noqa: E402


def main():
    torch.manual_seed(0)
    B, T, Hv, Wv, A = 1, 15, 4, 4, 60
    video = torch.randn(B, 24, T, Hv, Wv)
    audio = torch.randn(B, 32, 2, A)  # the fitted driving-track latent
    fps = 24.0
    frames = chain._frame_count(T)
    duration_s = frames / fps
    steps_per_second = A / duration_s

    locked_video_mask = torch.ones(B, 1, T, Hv, Wv)
    locked_audio_mask = torch.zeros(B, 1, 2, A)
    locked_latent = {
        "samples": chain.comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": chain.comfy.nested_tensor.NestedTensor(
            (locked_video_mask, locked_audio_mask)
        ),
    }

    # 1) window covering the first half of the clip, default margins/ramp
    node = chain.MiniMaxH3AudioWindowLock
    (out,) = node.apply(
        locked_latent,
        speech_start_s=0.0,
        speech_end_s=duration_s / 2,
        fps=fps,
        head_margin_s=0.2,
        tail_margin_s=0.4,
        ramp_steps=2,
    )
    out_video, out_audio = out["samples"].tensors
    out_vmask, out_amask = out["noise_mask"].tensors
    lo = int(min(A, max(0, round((0.0 - 0.2) * steps_per_second))))
    hi = int(min(A, max(0, round((duration_s / 2 + 0.4) * steps_per_second))))
    # 1) audio stream: verbatim inside the locked core window, zeroed outside
    assert torch.equal(out_audio[..., lo:hi], audio[..., lo:hi]), "locked window audio was mutated"
    if lo > 0:
        assert float(out_audio[..., :lo].abs().max()) == 0.0, "head reference audio was not zeroed"
    assert float(out_audio[..., hi:].abs().max()) == 0.0, "tail reference audio was not zeroed"
    assert torch.equal(out_video, video), "video stream was mutated"
    assert torch.equal(out_vmask, locked_video_mask), "video mask was mutated"

    assert float(out_amask[..., lo:hi].max()) == 0.0, "speech window is not locked"
    assert float(out_amask[..., hi + 2:].min()) == 1.0, "tail is not unlocked"
    ramp = out_amask[..., hi:hi + 2]
    assert float(ramp.min()) > 0.0 and float(ramp.max()) <= 1.0, "tail ramp missing"
    flat_ramp = ramp.flatten()
    assert float(flat_ramp[0]) < float(flat_ramp[-1]), "tail ramp is not increasing"
    # head: window starts at 0 and the head margin is clipped, so no head ramp
    if lo > 0:
        assert float(out_amask[..., :lo].min()) == 1.0, "unexpected head ramp"

    # 2) margins grow the locked window: tail margin 0 fully unlocks from speech end
    (out2,) = node.apply(
        locked_latent, 0.0, duration_s / 2, fps=fps,
        head_margin_s=0.0, tail_margin_s=0.0, ramp_steps=0,
    )
    _, amask2 = out2["noise_mask"].tensors
    hi2 = int(min(A, max(0, round(duration_s / 2 * steps_per_second))))
    assert float(amask2[..., :hi2].max()) == 0.0, "speech window moved without margins"
    assert float(amask2[..., hi2:].min()) == 1.0, "no-margin window still locks the tail"

    # 3) speech_end beyond the clip: stays fully locked (pass-through semantics)
    (out3,) = node.apply(
        locked_latent, 0.0, duration_s + 5.0, fps=fps,
        head_margin_s=0.0, tail_margin_s=0.0, ramp_steps=0,
    )
    _, amask3 = out3["noise_mask"].tensors
    assert float(amask3.max()) == 0.0, "over-long speech window must stay fully locked"
    out_video3, out_audio3 = out3["samples"].tensors
    assert torch.equal(out_audio3, audio), "fully-locked pass-through must not touch the audio stream"

    # 4) TakeGuide compatibility: windowed mask still counts as an audio lock
    #    and survives the seam verbatim (audio stream untouched, no carry pin).
    take = chain.H3Take(
        video=torch.randn(B, 24, 20, Hv, Wv),
        audio=torch.randn(B, 32, 2, A),
        meta={"frames": 22},
    )
    cond, lat, _ = chain.MiniMaxH3TakeGuide.guide(
        [[torch.zeros(1), {}]], out, 22, take=take, frame_idx=0,
        mode="guide", audio_mode="carry",
    )
    _, seam_audio = lat["samples"].tensors
    _, seam_amask = lat["noise_mask"].tensors
    assert torch.equal(seam_audio, out_audio), "TakeGuide carry overwrote the windowed take's audio"
    assert torch.equal(seam_amask, out_amask), "TakeGuide altered the windowed audio mask"

    # 5) a latent without a noise_mask is rejected loudly
    try:
        node.apply({"samples": chain.comfy.nested_tensor.NestedTensor((video, audio))}, 0.0, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("unlocked latent must be rejected, not silently windowed")

    # 6) freeze_video=True pins the entire video stream (two-pass audio refine)
    (out6,) = node.apply(
        locked_latent, 0.0, duration_s / 2, fps=fps,
        head_margin_s=0.2, tail_margin_s=0.4, ramp_steps=2, freeze_video=True,
    )
    video6, _ = out6["samples"].tensors
    vmask6, amask6 = out6["noise_mask"].tensors
    assert float(vmask6.max()) == 0.0, "freeze_video must zero the whole video mask"
    assert torch.equal(video6, video), "video stream content must be untouched by freeze_video"
    assert float(amask6[..., hi + 2:].min()) == 1.0, "audio tail must stay unlocked in freeze mode"

    # 7) multi-span mode: two lines, generated head / gap / tail
    d = duration_s
    m = 0.02 * d
    spans = json.dumps([[0.15 * d, 0.45 * d], [0.55 * d, 0.9 * d]])
    (out7,) = node.apply(
        locked_latent, 0.0, 0.0, fps=fps,
        head_margin_s=m, tail_margin_s=m, ramp_steps=0,
        speech_spans=spans,
    )
    out_audio7 = out7["samples"].tensors[1]
    _, amask7 = out7["noise_mask"].tensors
    step = steps_per_second

    def seg(a, b):  # (lo, hi) step indices for a second-range
        return max(0, int(round(a * step))), min(A, int(round(b * step)))

    s1_lo, s1_hi = seg(0.15 * d - m, 0.45 * d + m)
    s2_lo, s2_hi = seg(0.55 * d - m, 0.9 * d + m)
    head_end = seg(0, 0.15 * d - m)[1]
    gap_lo, _ = seg(0.45 * d + m, 0.55 * d - m)
    assert float(amask7[..., s1_lo:s1_hi].max()) == 0.0, "span 1 is not locked"
    assert float(amask7[..., s2_lo:s2_hi].max()) == 0.0, "span 2 is not locked"
    assert float(amask7[..., :head_end].min()) == 1.0, "head is not unlocked"
    assert float(amask7[..., gap_lo:s2_lo].min()) == 1.0, "gap is not unlocked"
    assert float(amask7[..., s2_hi:].min()) == 1.0, "tail is not unlocked"
    # audio stream verbatim inside both spans, zeroed in head/gap/tail
    assert torch.equal(out_audio7[..., s1_lo:s1_hi], audio[..., s1_lo:s1_hi]), "span 1 audio mutated"
    assert torch.equal(out_audio7[..., s2_lo:s2_hi], audio[..., s2_lo:s2_hi]), "span 2 audio mutated"
    assert float(out_audio7[..., :s1_lo].abs().max()) == 0.0, "head audio not zeroed"
    assert float(out_audio7[..., s1_hi:s2_lo].abs().max()) == 0.0, "gap audio not zeroed"
    assert float(out_audio7[..., s2_hi:].abs().max()) == 0.0, "tail audio not zeroed"

    # 8) margins that bridge a small gap merge the spans into one lock
    spans8 = json.dumps([[0.15 * d, 0.45 * d], [0.55 * d, 0.9 * d]])
    (out8,) = node.apply(
        locked_latent, 0.0, 0.0, fps=fps,
        head_margin_s=0.06 * d, tail_margin_s=0.06 * d, ramp_steps=0,
        speech_spans=spans8,
    )
    _, amask8 = out8["noise_mask"].tensors
    assert float(amask8[..., s1_lo:s2_hi].max()) == 0.0, "margin-bridged spans must merge into one lock"
    assert float(amask8[..., :seg(0, 0.15 * d - 0.06 * d)[1]].min()) == 1.0, "head must stay unlocked after merge"
    assert float(amask8[..., seg(0.9 * d + 0.06 * d, d)[0]:].min()) == 1.0, "tail must stay unlocked after merge"

    # 9) malformed speech_spans is rejected loudly
    for bad in ("{not json", "[]", "[[1]]", "[[3, 1]]"):
        try:
            node.apply(locked_latent, 0.0, 0.0, fps=fps, speech_spans=bad)
        except ValueError:
            continue
        raise AssertionError(f"speech_spans={bad!r} must be rejected")

    print("OK: windowed audio lock built; locked window verbatim, outside zeroed; seam-compatible")


if __name__ == "__main__":
    main()