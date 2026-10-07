"""H3 LatentChain: portable AV takes on disk for seamless long-form chaining.

Units are single safetensors files under ``output/h3_chain/<session>/`` holding
the two raw H3 latent streams plus a JSON header (geometry, frame grid).
Everything here speaks only core H3 types: the AV nested-tensor latent and the
``minimax_keyframes`` conditioning slot defined by
``comfy_extras/nodes_minimax_h3.py``. Nothing depends on a companion node
pack; the optional segments output is only a compatibility shim for
ComfyUI-MiniMaxH3-Easy's ``prev_segments`` consumers.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

import folder_paths
import node_helpers
import comfy.nested_tensor
from comfy.ldm.minimax.model import FRAME_PER_TOKEN

FORMAT = "h3-latent-chain"
FORMAT_VERSION = 1
CHAIN_DIR = "h3_chain"
TAKE_TYPE = "H3_TAKE"
SEGMENTS_TYPE = "MINIMAX_H3_SEGMENTS"
GUIDE_HANDOFF_VIDEO_TOKENS = 2
HEADER_KEY = "h3_chain"
SESSION_NONE = "(no session)"
UNIT_NONE = "(no unit)"

try:  # core helper module ships with every H3-capable ComfyUI
    from comfy_extras.nodes_minimax_h3 import temporal_shape as _temporal_shape
except Exception:  # pragma: no cover - very old cores
    _temporal_shape = None


# ---------------------------------------------------------------------------
# takes and files
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class H3Take:
    """One rendered clip's AV latents plus the header that describes them."""

    video: torch.Tensor  # [B,24,T,H/16,W/16]
    audio: torch.Tensor  # [B,32,2,N]
    meta: Mapping[str, Any] = field(default_factory=dict)


def _temporal_shape_or_local(length: int) -> tuple[int, int, int]:
    if _temporal_shape is not None:
        return _temporal_shape(length)
    frame_count = max(5, int(length))
    while frame_count % 17 != 5:
        frame_count += 1
    if frame_count <= 5:
        latent_t = 2
    else:
        latent_t = ((frame_count - 5) // 17) * 5 + 2
    return frame_count, latent_t, round(frame_count / 24.0 * 40)


def _split_streams(latent: Any, message: str) -> tuple[torch.Tensor, torch.Tensor]:
    samples = latent.get("samples") if isinstance(latent, Mapping) else None
    if hasattr(samples, "unbind"):
        streams = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        streams = list(samples)
    else:
        raise ValueError(message)
    if (
        len(streams) < 2
        or not all(isinstance(stream, torch.Tensor) for stream in streams[:2])
        or streams[0].ndim != 5
        or int(streams[0].shape[1]) != 24
        or streams[1].ndim != 4
    ):
        raise ValueError(message)
    return streams[0], streams[1]


def _pack_latent(video: torch.Tensor, audio: torch.Tensor) -> dict[str, Any]:
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _pack_latent_masks(
    video: torch.Tensor,
    audio: torch.Tensor,
    video_mask: torch.Tensor,
    audio_mask: torch.Tensor,
) -> dict[str, Any]:
    latent = _pack_latent(video, audio)
    latent["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return latent


def _frame_count(video_tokens: int) -> int:
    return sum(int(FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)]) for index in range(int(video_tokens)))


def _chain_roots() -> list[str]:
    # input/h3_chain first: farm jobs upload takes as stateless input assets;
    # output/h3_chain keeps local same-session chaining working with zero setup.
    return [
        os.path.join(folder_paths.get_input_directory(), CHAIN_DIR),
        os.path.join(folder_paths.get_output_directory(), CHAIN_DIR),
    ]


def _output_root() -> str:
    return os.path.join(folder_paths.get_output_directory(), CHAIN_DIR)


def _safe_name(value: Any, fallback: str) -> str:
    name = str(value or "").strip().replace("\\", "/").split("/")[-1]
    return name or fallback


def _session_dir(session: str) -> str:
    session = _safe_name(session, "")
    for root in _chain_roots():
        path = os.path.normpath(os.path.join(root, session))
        if path.startswith(os.path.normpath(root) + os.sep) and os.path.isdir(path):
            return path
    raise ValueError(f"LatentChain session folder not found: {session}")


def _list_sessions() -> list[str]:
    found = []
    for root in _chain_roots():
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if os.path.isdir(os.path.join(root, name)) and _safe_name(name, "") and name not in found:
                found.append(name)
    return sorted(found)


def _list_units(session: str) -> list[str]:
    try:
        path = _session_dir(session)
    except ValueError:
        return []
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(path)
        if name.endswith((".safetensors", ".pt")) and os.path.isfile(os.path.join(path, name))
    )


def _load_take(session: str, unit: str) -> H3Take:
    folder = _session_dir(session)
    base = _safe_name(unit, "")
    for suffix in (".safetensors", ".pt"):
        path = os.path.join(folder, base + suffix)
        if os.path.isfile(path):
            break
    else:
        raise ValueError(f"LatentChain take not found: {session}/{base}")
    if path.endswith(".pt"):
        # Legacy ComfyUI-MiniMaxH3-Easy latent-bridge file: {video_latent,
        # audio_latent, meta{width,height,fps,...}} written by torch.save.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = dict(payload.get("meta") or {})
        video = payload["video_latent"]
        audio = payload["audio_latent"]
        meta.setdefault("frames", _frame_count(int(video.shape[2])))
        meta.setdefault("width", int(video.shape[4]) * 16)
        meta.setdefault("height", int(video.shape[3]) * 16)
        meta.setdefault("fps", 24.0)
        return H3Take(video=video, audio=audio, meta=meta)
    from safetensors.torch import load_file

    tensors = load_file(path)
    with open(path, "rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len).decode("utf-8"))
    meta = json.loads((header.get("__metadata__") or {}).get(HEADER_KEY, "{}"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"Not a LatentChain file: {path}")
    video = tensors.get("video")
    audio = tensors.get("audio")
    if video is None or audio is None:
        raise ValueError(f"LatentChain file is missing its AV streams: {path}")
    return H3Take(video=video, audio=audio, meta=meta)


def _easy_segments(take: H3Take):
    """Wrap a take as an Easy segment result when that pack is installed."""
    result_cls = sample_cls = None
    for module in list(sys.modules.values()):
        # torch module __getattr__ fabricates objects for unknown names
        result_cls = module.__dict__.get("MiniMaxH3SegmentResult")
        sample_cls = module.__dict__.get("MiniMaxH3SegmentSample")
        if (
            isinstance(result_cls, type)
            and isinstance(sample_cls, type)
            and result_cls.__name__ == "MiniMaxH3SegmentResult"
        ):
            break
    if (
        not isinstance(result_cls, type)
        or not isinstance(sample_cls, type)
        or result_cls.__name__ != "MiniMaxH3SegmentResult"
    ):
        print("[H3 LatentChain] segments output skipped: ComfyUI-MiniMaxH3-Easy not found")
        return None
    frames = int(take.meta.get("frames") or _frame_count(int(take.video.shape[2])))
    sample = sample_cls(
        video_latent=take.video.detach().to("cpu").contiguous(),
        audio_latent=take.audio.detach().to("cpu").contiguous(),
        head_frames=0,
        delivery_frames=frames,
        output_frames=frames,
    )
    plan = {
        "width": int(take.meta.get("width") or int(take.video.shape[4]) * 16),
        "height": int(take.meta.get("height") or int(take.video.shape[3]) * 16),
        "fps": float(take.meta.get("fps") or 24.0),
        "continuity_mode": "chain",
        "latent_bridge": True,
    }
    return result_cls(plan=plan, samples=(sample,))


# ---------------------------------------------------------------------------
# guide math (ported payload shapes from the Easy context-segment helpers)
# ---------------------------------------------------------------------------

def _tail_tokens(video: torch.Tensor, tokens: int) -> torch.Tensor:
    available = int(video.shape[2])
    if available >= tokens:
        return video[:, :, -tokens:]
    return torch.cat(
        [video[:, :, :1].repeat(1, 1, tokens - available, 1, 1), video], dim=2
    )


def _context_audio_window(audio: torch.Tensor, steps: int) -> torch.Tensor:
    available = int(audio.shape[-1])
    if available >= steps:
        return audio[..., -steps:]
    return torch.cat(
        [audio[..., :1].repeat(1, 1, 1, steps - available), audio], dim=-1
    )


def _fit_window(window: torch.Tensor, steps: int) -> torch.Tensor:
    value = window[..., :steps]
    if int(value.shape[-1]) < steps:
        value = torch.cat(
            [value, value[..., -1:].repeat(1, 1, 1, steps - int(value.shape[-1]))], dim=-1
        )
    return value


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------

class MiniMaxH3ChainSave:
    """Write a sampled H3 AV latent to disk as a reusable take."""

    CATEGORY = "latent/chain"
    FUNCTION = "save"
    RETURN_TYPES = ("LATENT", TAKE_TYPE, "INT")
    RETURN_NAMES = ("latent", "take", "frames")
    DESCRIPTION = (
        "Store the sampled MiniMax H3 video+audio latent as a safetensors take "
        "(output/h3_chain/<session>/). Passes the latent through unchanged so "
        "it can still feed the decoders in the same graph."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Sampled MiniMax H3 AV latent (KSampler / SamplerCustomAdvanced output)."}),
                "session": ("STRING", {"default": "session_a", "tooltip": "Session folder under output/h3_chain."}),
                "name": ("STRING", {"default": "take_001", "tooltip": "Unit name (one .safetensors file)."}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
            },
        }

    @classmethod
    def save(cls, latent, session, name, fps=24.0):
        video, audio = _split_streams(latent, "LatentChain Save expects a MiniMax H3 AV latent")
        video = video.detach().to("cpu").contiguous()
        audio = audio.detach().to("cpu").contiguous()
        frames = _frame_count(int(video.shape[2]))
        meta = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "model": "minimax_h3",
            "width": int(video.shape[4]) * 16,
            "height": int(video.shape[3]) * 16,
            "fps": float(fps),
            "frames": frames,
            "video_tokens": int(video.shape[2]),
            "audio_steps": int(audio.shape[-1]),
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        session = _safe_name(session, "session")
        name = _safe_name(name, "take")
        folder = os.path.join(_output_root(), session)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name + ".safetensors")
        from safetensors.torch import save_file

        save_file(
            {"video": video, "audio": audio},
            path,
            metadata={HEADER_KEY: json.dumps(meta)},
        )
        print(f"[H3 LatentChain] take saved: {path}")
        return (latent, H3Take(video=video, audio=audio, meta=meta), frames)


class MiniMaxH3ChainLoad:
    """Load a saved take as a chaining source for a new clip."""

    CATEGORY = "latent/chain"
    FUNCTION = "load"
    RETURN_TYPES = (TAKE_TYPE, "INT", "INT", "FLOAT", SEGMENTS_TYPE)
    RETURN_NAMES = ("take", "width", "height", "fps", "segments")
    DESCRIPTION = (
        "Read one LatentChain unit into a take handle for the Take Guide. "
        "The segments output (needs ComfyUI-MiniMaxH3-Easy) plugs into "
        "prev_segments-style consumers as a whole delivered take."
    )

    @classmethod
    def INPUT_TYPES(cls):
        sessions = _list_sessions()
        units: list[str] = []
        for session in sessions:
            for unit in _list_units(session):
                if unit not in units:
                    units.append(unit)
        return {
            "required": {
                "session": (sessions + [SESSION_NONE], {"default": sessions[0] if sessions else SESSION_NONE, "tooltip": f"Session folder under output/h3_chain. {SESSION_NONE} = no history yet (first clip of a new session). Refresh the page to re-scan."}),
                "name": (units + [UNIT_NONE], {"default": units[0] if units else UNIT_NONE, "tooltip": f"Take unit inside the session. {UNIT_NONE} = no history yet. Refresh the page to re-scan."}),
            },
            "optional": {
                "missing_take": (["empty", "error"], {"default": "empty", "tooltip": "empty: emit no take (pass-through) when the unit is not on disk yet; error: fail the run."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        stamp = []
        for root in _chain_roots():
            if not os.path.isdir(root):
                continue
            for session in os.listdir(root):
                folder = os.path.join(root, session)
                if os.path.isdir(folder):
                    stamp.append(f"{os.path.basename(os.path.dirname(root))}/{session}:{os.path.getmtime(folder)}")
        return "|".join(stamp) or "empty"

    @classmethod
    def load(cls, session, name, missing_take="empty"):
        if session == SESSION_NONE or name == UNIT_NONE:
            return (None, 0, 0, 24.0, None)
        try:
            take = _load_take(session, name)
        except ValueError:
            if missing_take == "empty":
                print(f"[H3 LatentChain] take {session}/{name} not on disk yet: pass-through")
                return (None, 0, 0, 24.0, None)
            raise
        frames = int(take.meta.get("frames") or _frame_count(int(take.video.shape[2])))
        width = int(take.meta.get("width") or int(take.video.shape[4]) * 16)
        height = int(take.meta.get("height") or int(take.video.shape[3]) * 16)
        fps = float(take.meta.get("fps") or 24.0)
        segments = _easy_segments(take)
        return (take, width, height, fps, segments)


class MiniMaxH3TakeGuide:
    """Seam a new clip onto a saved take in latent space (no RGB round trip).

    Guide mode (default) anchors the take's tail as a frame-index keyframe
    (exactly the official MiniMaxH3AddGuide payload, straight from disk) and
    pins only the boundary video tokens plus the full audio overlap into the
    fresh latent's denoise mask, so the model redraws the repeated head and
    recursive degradation is avoided. av_prefix pins the whole overlap.
    audio_mode="fresh" skips the audio pin entirely: the new take denoises
    its own wav-conditioned audio from step 0 (video handoff only).
    """

    CATEGORY = "latent/chain"
    FUNCTION = "guide"
    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "latent", "covered_frames")
    DESCRIPTION = (
        "Anchor a loaded take's latents at a frame of the new clip. Insert "
        "between the H3 conditioning node and the guider/sampler. "
        "covered_frames head frames of the new clip are consumed as the "
        "seam; subtract them from the desired output length."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "New clip's conditioning (MiniMaxH3ImageToVideo / ReferenceToVideo)."}),
                "latent": ("LATENT", {"tooltip": "New clip's fresh AV latent from the same node."}),
                "context": ("INT", {"default": 22, "min": 5, "max": 362, "step": 17, "tooltip": "Boundary history in pixel frames (clips snap to 17k+5)."}),
            },
            "optional": {
                "take": (TAKE_TYPE, {"tooltip": "Take handle from LatentChain Load. Unconnected/empty = pass-through (first clip of a session)."}),
                "frame_idx": ("INT", {"default": 0, "min": -9999, "max": 9999, "tooltip": "Frame to anchor the take's tail at. Negative counts from the end. Seam handoff only applies at frame 0."}),
                "mode": (["guide", "av_prefix"], {"default": "guide"}),
                "audio_mode": (["carry", "fresh"], {"default": "carry", "tooltip": "carry: pin the previous take's audio into the new latent head (legacy). fresh: video-only handoff — the new take keeps its own wav-conditioned audio timeline (lip sync stays frame-exact)."}),
            },
        }

    @classmethod
    def guide(cls, positive, latent, context, take=None, frame_idx=0, mode="guide", audio_mode="carry"):
        if take is None:
            return (positive, latent, 0)
        if not isinstance(take, H3Take):
            raise ValueError("Take Guide expects a MiniMaxH3ChainLoad take")
        new_video, new_audio = _split_streams(latent, "Take Guide expects a MiniMax H3 AV latent")
        ref_video, ref_audio = take.video, take.audio
        if (
            int(ref_video.shape[3]) != int(new_video.shape[3])
            or int(ref_video.shape[4]) != int(new_video.shape[4])
        ):
            raise ValueError(
                f"Take resolution {int(ref_video.shape[4]) * 16}x{int(ref_video.shape[3]) * 16} "
                f"does not match the new clip {int(new_video.shape[4]) * 16}x{int(new_video.shape[3]) * 16}"
            )
        context = max(5, int(context))
        while context % 17 != 5 and context > 5:
            context -= 1
        _frames, prefix_tokens, audio_window = _temporal_shape_or_local(context)
        new_frames = _frame_count(int(new_video.shape[2]))
        resolved = int(frame_idx) if int(frame_idx) >= 0 else new_frames + int(frame_idx)
        if resolved < 0 or resolved >= new_frames:
            raise ValueError(f"frame_idx {frame_idx} is outside the new clip's {new_frames} frames")

        # conditioning: anchor the take tail (evict any keyframe at the same slot)
        tail = _tail_tokens(ref_video, prefix_tokens).detach().to("cpu").contiguous()
        keyframes = [
            kf for kf in list(positive[0][1].get("minimax_keyframes", []))
            if int(kf.get("resolved_frame_index", -1)) != resolved
        ]
        evicted = len(positive[0][1].get("minimax_keyframes", [])) - len(keyframes)
        if evicted:
            print(f"[H3 LatentChain] Take Guide replaced {evicted} keyframe(s) at frame {resolved}")
        keyframes.append({"resolved_frame_index": resolved, "latent": tail})
        conditioning = node_helpers.conditioning_set_values(
            positive, {"minimax_keyframes": keyframes}
        )
        covered = _frame_count(int(tail.shape[2]))

        new_video = new_video.clone()
        new_audio = new_audio.clone()
        video_mask = torch.ones(
            new_video.shape[:1] + (1,) + new_video.shape[2:],
            device=new_video.device, dtype=new_video.dtype,
        )
        audio_mask = torch.ones(
            new_audio.shape[:1] + (1,) + new_audio.shape[2:],
            device=new_audio.device, dtype=new_audio.dtype,
        )

        if resolved == 0:
            # latent handoff: pin the boundary so phase and motion direction survive
            video_prefix = min(int(new_video.shape[2]), int(prefix_tokens))
            if video_prefix > 0:
                window = _tail_tokens(ref_video, video_prefix)
                if mode == "av_prefix":
                    edge_start = 0
                else:
                    edge_start = max(0, video_prefix - min(GUIDE_HANDOFF_VIDEO_TOKENS, video_prefix))
                new_video[:, :, edge_start:video_prefix] = window[:, :, edge_start:video_prefix].to(
                    device=new_video.device, dtype=new_video.dtype
                )
                video_mask[:, :, edge_start:video_prefix] = 0.0
            if audio_mode == "carry":
                audio_prefix = min(int(new_audio.shape[-1]), int(audio_window))
                if audio_prefix > 0:
                    window = _fit_window(_context_audio_window(ref_audio, audio_window), audio_prefix)
                    new_audio[..., :audio_prefix] = window.to(device=new_audio.device, dtype=new_audio.dtype)
                    audio_mask[..., :audio_prefix] = 0.0
            # "fresh": video-only handoff — leave new_audio / audio_mask untouched so
            # the take's own wav-conditioned audio latent denoises from step 0 and the
            # mouth stays on its conditioning-wav grid.

        if bool(torch.any(video_mask < 1.0)) or bool(torch.any(audio_mask < 1.0)):
            packed = _pack_latent_masks(new_video, new_audio, video_mask, audio_mask)
        else:
            packed = _pack_latent(new_video, new_audio)
        return (conditioning, packed, covered)


class MiniMaxH3KeyframeRescale:
    """Rescale chain keyframe anchors to a downstream latent resolution.

    Two-stage graphs (e.g. the 7+1 split-sampling path) refine the upscaled
    latent in a second pass at 2x resolution. TakeGuide anchors the previous
    take's tail as a ``minimax_keyframes`` latent at the FIRST-pass
    resolution; the second pass must see anchors that match its own token
    grid, or the model's keyframe insertion fails with a token-shape broadcast
    error (stage-1-sized anchor tokens vs stage-2 anchor slots). This node
    spatially interpolates every keyframe video latent to match an optional
    reference AV latent (wire the second stage's packed AV latent), leaving
    audio keyframes and all other conditioning values untouched. The input
    conditioning is never mutated, so the same TakeGuide output can still feed
    the first stage unrescaled.
    """

    CATEGORY = "latent/chain"
    FUNCTION = "rescale"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    DESCRIPTION = (
        "Match chain keyframe anchor latents to a second-stage latent "
        "resolution (two-stage graphs need the stage-2 guider to see "
        "anchors at ITS resolution)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "Take Guide's positive conditioning."}),
            },
            "optional": {
                "reference_latent": ("LATENT", {"tooltip": "The latent the anchors must match: the second stage's packed AV latent. Required when keyframes are present."}),
                "mode": (["bilinear", "bicubic", "nearest", "drop"], {"default": "bilinear", "tooltip": "Spatial interpolation for the anchor latents. drop: remove ALL minimax_keyframes from this branch (stage 2 then treats the pinned head like any other token - use when a rescaled anchor conflicts with the pinned values and ghosting appears)."}),
            },
        }

    @classmethod
    def rescale(cls, positive, reference_latent=None, mode="bilinear"):
        target = None
        if reference_latent is not None:
            try:
                video, _ = _split_streams(
                    reference_latent,
                    "Keyframe Rescale expects an H3 AV latent as reference")
            except ValueError:
                samples = reference_latent.get("samples") if isinstance(reference_latent, Mapping) else None
                if isinstance(samples, torch.Tensor) and samples.ndim == 5:
                    video = samples
                else:
                    raise ValueError(
                        "Keyframe Rescale reference must be an H3 AV latent "
                        "(packed AV or plain video latent)")
            target = (int(video.shape[-2]), int(video.shape[-1]))
        out = []
        changed = 0
        dropped = 0
        for embedding, metadata in positive:
            values = dict(metadata)
            keyframes = values.get("minimax_keyframes")
            if mode == "drop":
                if keyframes:
                    values.pop("minimax_keyframes", None)
                    dropped += len(keyframes)
                out.append([embedding, values])
                continue
            if not keyframes:
                out.append([embedding, values])
                continue
            if target is None:
                raise ValueError(
                    "Keyframe Rescale: keyframes present but no "
                    "reference_latent wired - anchor sizes cannot be matched")
            new_keyframes = []
            for kf in keyframes:
                kf = dict(kf)
                lat = kf.get("latent")
                if isinstance(lat, torch.Tensor) and lat.ndim == 5 and target is not None:
                    if (int(lat.shape[-2]), int(lat.shape[-1])) != target:
                        # spatial-only upscale: flatten (N, C, T) -> (N, C*T)
                        # so F.interpolate never touches the temporal axis
                        n, c, t = int(lat.shape[0]), int(lat.shape[1]), int(lat.shape[2])
                        flat = lat.reshape(n, c * t, int(lat.shape[-2]), int(lat.shape[-1]))
                        kwargs = {} if mode == "nearest" else {"align_corners": False}
                        up = torch.nn.functional.interpolate(
                            flat, size=target, mode=mode, **kwargs)
                        kf["latent"] = up.reshape(
                            n, c, t, int(target[0]), int(target[1])).contiguous()
                        changed += 1
                new_keyframes.append(kf)
            values["minimax_keyframes"] = new_keyframes
            out.append([embedding, values])
        if changed:
            print(f"[H3 LatentChain] Keyframe Rescale: resized {changed} anchor "
                  f"latent(s) to {target[1]}x{target[0]} ({mode})")
        if dropped:
            print(f"[H3 LatentChain] Keyframe Rescale: dropped {dropped} "
                  f"anchor(s) from this branch (mode=drop)")
        return (out,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ChainSave": MiniMaxH3ChainSave,
    "MiniMaxH3ChainLoad": MiniMaxH3ChainLoad,
    "MiniMaxH3TakeGuide": MiniMaxH3TakeGuide,
    "MiniMaxH3KeyframeRescale": MiniMaxH3KeyframeRescale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ChainSave": "MiniMax H3 Chain Save",
    "MiniMaxH3ChainLoad": "MiniMax H3 Chain Load",
    "MiniMaxH3TakeGuide": "MiniMax H3 Take Guide (Seam)",
    "MiniMaxH3KeyframeRescale": "MiniMax H3 Keyframe Rescale (Stage 2)",
}
