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


def _load_take_from_path(path: str) -> H3Take:
    """Parse one take file from its full path (shared by session and file loaders)."""
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


def _load_take(session: str, unit: str) -> H3Take:
    folder = _session_dir(session)
    base = _safe_name(unit, "")
    for suffix in (".safetensors", ".pt"):
        path = os.path.join(folder, base + suffix)
        if os.path.isfile(path):
            break
    else:
        raise ValueError(f"LatentChain take not found: {session}/{base}")
    return _load_take_from_path(path)


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


def _resolve_uploaded_take_file(latent_filename: str) -> str:
    """Resolve an uploaded take file by its input-directory basename.

    Platform jobs upload takes as stateless input assets (balancer fanout
    lands them flat in the input dir, content-hash named), so they are
    addressed by filename instead of the session/name combo pair. Only a
    bare basename is accepted (no subdirs, no traversal); only extensions
    the take parser understands (.safetensors / .pt) resolve. Returns ""
    when the name is absent or the file is not on disk yet (caller decides
    pass-through vs error).
    """
    name = str(latent_filename or "").strip().replace("\\", "/")
    if not name or os.path.basename(name) != name:
        return ""
    ext = os.path.splitext(name)[1].lower()
    if ext not in (".safetensors", ".pt", ".latent"):
        return ""
    path = os.path.join(folder_paths.get_input_directory(), name)
    return path if os.path.isfile(path) else ""


class MiniMaxH3ChainLoadFromFile:
    """Load a take from an uploaded input-dir file (free-string addressing).

    Same output tuple as ChainLoad, but the take is named by a plain STRING
    instead of disk-scanned combo enums, so workflow-as-model jobs (Jellyfish
    /api/v1/film/tasks/video) can reference takes uploaded per-task that no
    enum could ever list. Empty string = no history (identical to
    ChainLoad's (no session)/(no unit) sentinels).
    """

    CATEGORY = "latent/chain"
    FUNCTION = "load"
    RETURN_TYPES = (TAKE_TYPE, "INT", "INT", "FLOAT", SEGMENTS_TYPE)
    RETURN_NAMES = ("take", "width", "height", "fps", "segments")
    DESCRIPTION = (
        "Load one LatentChain unit from an uploaded input file (free-string "
        "filename, no combo scan). Empty = no history yet. Drop-in for "
        "ChainLoad in API-driven graphs where the take arrives as an upload."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # STRING, not combo: the file is uploaded per job (content-hash
                # named), so it can never appear in a static enum list.
                "latent_filename": ("STRING", {"default": "", "tooltip": "Take file in the input dir (.safetensors/.pt). Empty = no history (head take)."}),
            },
            "optional": {
                "missing_take": (["empty", "error"], {"default": "empty", "tooltip": "empty: emit no take (pass-through) when the file is not uploaded yet; error: fail the run."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, latent_filename, missing_take="empty"):
        path = _resolve_uploaded_take_file(latent_filename)
        return str(os.path.getmtime(path)) if path else "empty"

    @classmethod
    def load(cls, latent_filename, missing_take="empty"):
        path = _resolve_uploaded_take_file(latent_filename)
        if not path:
            if missing_take == "error" and str(latent_filename or "").strip():
                raise ValueError(f"LatentChain take file not found in input dir: {latent_filename}")
            return (None, 0, 0, 24.0, None)
        try:
            take = _load_take_from_path(path)
        except ValueError:
            if missing_take == "empty":
                print(f"[H3 LatentChain] take file {latent_filename} failed to load: pass-through")
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
                "audio_mode": (["carry", "fresh"], {"default": "carry", "tooltip": "carry: pin the previous take's audio into the new latent head (legacy). fresh: video-only handoff - the new take keeps its own wav-conditioned audio timeline (lip sync stays frame-exact). An incoming audio lock (digital_human driving track) is preserved verbatim in both modes; carry is skipped for locked takes."}),
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

        # Detect an incoming audio lock (MiniMaxH3Easy digital_human pins the
        # driving track into the target audio latent with a zero denoise mask)
        # BEFORE building anchors — a locked (dialogue) take needs a different
        # head contract than a generated-audio chain.
        incoming_mask = latent.get("noise_mask") if isinstance(latent, Mapping) else None
        incoming_audio_mask = None
        if incoming_mask is not None:
            tensors = getattr(incoming_mask, "tensors", None)
            if isinstance(tensors, (tuple, list)) and len(tensors) >= 2:
                candidate = tensors[1]
                expected_shape = tuple(new_audio.shape[:1] + (1,) + new_audio.shape[2:])
                if (
                    isinstance(candidate, torch.Tensor)
                    and tuple(candidate.shape) == expected_shape
                    and bool(torch.any(candidate < 1.0))
                ):
                    incoming_audio_mask = candidate
        audio_locked = incoming_audio_mask is not None

        # conditioning: anchor the take tail (evict any keyframe at the same slot).
        #
        # Head-anchor contract differs by audio state (2026-10-14, s03 t06 lip
        # suppression): anchoring the full prefix_tokens tail (~1.2s) at frame 0
        # freezes the model into "continuation of the static tail" for that whole
        # window — speech starting at second 0 gets no articulation (measured:
        # "House rules new member" 0-1.27s rendered with a closed mouth while
        # "loser strips ..." from 1.57s tracked frame-exact). For a LOCKED
        # dialogue take the seam must behave like a head take's first-frame
        # keyframe: anchor ONE token (~4 frames) so the cut stays visually
        # seamless, then leave the head free to articulate with the locked
        # driving track. Unlocked chains keep the full-tail anchor (their
        # continuity relies on it).
        anchor_tokens = 1 if audio_locked else prefix_tokens
        tail = _tail_tokens(ref_video, anchor_tokens).detach().to("cpu").contiguous()
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
        # Preserve the incoming audio lock (detected above): the audio stream
        # and mask pass through verbatim; carry is skipped (the per-take mux
        # owns the final soundtrack, so generated-audio carry across the seam
        # is meaningless for a locked track).
        video_mask = torch.ones(
            new_video.shape[:1] + (1,) + new_video.shape[2:],
            device=new_video.device, dtype=new_video.dtype,
        )
        if audio_locked:
            audio_mask = incoming_audio_mask.to(
                device=new_video.device, dtype=new_video.dtype
            )
        else:
            new_audio = new_audio.clone()
            audio_mask = torch.ones(
                new_audio.shape[:1] + (1,) + new_audio.shape[2:],
                device=new_audio.device, dtype=new_audio.dtype,
            )

        if resolved == 0:
            # latent handoff: pin the boundary so phase and motion direction
            # survive — but NOT on a locked dialogue take: the hard-frozen tail
            # edge is the second half of the head-stasis bracket (keyframe
            # anchor + pin) that suppressed speech starting at second 0
            # (s03 t06). A locked take's seam continuity is carried by the
            # 1-token keyframe anchor alone.
            if not audio_locked:
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
            # locked (or "fresh"): video-only handoff — the audio stream and
            # mask stay exactly as the conditioning node produced them (a locked
            # digital-human driving track must reach the sampler un-denoised).

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


class MiniMaxH3AudioWindowLock:
    """Unlock H3 audio generation outside a speech window on a dialogue take.

    MiniMaxH3Easy digital_human mode locks the ENTIRE audio timeline
    (``_lock_audio_latent``: audio_mask all zeros), so a dialogue take renders
    speech followed by dead room tone — H3 is forbidden from generating any
    ambience. This node rebuilds that lock into a windowed one: the audio mask
    stays 0 (keep the driving track verbatim) inside the speech window and
    becomes 1 (full denoise = H3 generates) everywhere else, so the same single
    render produces speech AND a generated soundscape. Video stream and video
    mask are untouched.

    The audio stream itself is ALSO rewritten: reference latent content outside
    the locked core window is zeroed. Easy's ``_fit_audio_latent`` pads a short
    track by repeating its LAST latent frame across the whole timeline tail, so
    an untouched reference tail feeds the sampler quasi-speech content in the
    to-be-generated region — at low sigmas the model sees that content and
    reconstructs the line (measured 2026-10-17: the dialogue line repeats in
    the unlocked tail on every seed and prompt variant). Zeroing makes the
    unlocked region start from the canonical empty audio state, so the model
    generates ambience instead of reconstructing speech.

    The mask lives on the audio latent grid (N steps spanning frames/fps
    seconds); the window is converted with margins and optional linear ramps at
    both edges. Downstream TakeGuide compatibility is preserved: the mask still
    contains values < 1, so the audio lock is detected and passed through
    verbatim (speech stays frame-exact).
    """

    CATEGORY = "latent/chain"
    FUNCTION = "apply"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    DESCRIPTION = (
        "Windowed audio lock for MiniMax H3 dialogue takes: keep the driving "
        "track verbatim inside the speech window, let H3 generate ambience "
        "everywhere else. Insert between Easy Output and Take Guide."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "MiniMax H3 AV latent from Easy Output (digital_human: audio locked)."}),
                "speech_start_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01, "tooltip": "Speech window start in seconds on the clip timeline."}),
                "speech_end_s": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 3600.0, "step": 0.01, "tooltip": "Speech window end in seconds. >= clip duration = stays fully locked."}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01, "tooltip": "Clip fps used to convert seconds to audio latent steps."}),
                "head_margin_s": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Extra locked seconds before the speech window (protects word onsets)."}),
                "tail_margin_s": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Extra locked seconds after the speech window (protects word tails)."}),
                "ramp_steps": ("INT", {"default": 2, "min": 0, "max": 40, "step": 1, "tooltip": "Linear ramp steps at both window edges (0 = hard cut). ~25ms per step at the 40Hz audio grid."}),
                "speech_spans": ("STRING", {"default": "", "tooltip": "Optional JSON list of [start_s, end_s] pairs for MULTIPLE dialogue lines (multi-span mode). When non-empty it replaces speech_start_s/speech_end_s; spans are expanded by the margins and merged when they overlap."}),
                "freeze_video": ("BOOLEAN", {"default": False, "tooltip": "Two-pass mode: set the VIDEO mask to 0 everywhere so the video stream cannot change at all (audio tail refinement against a finished picture)."}),
            },
        }

    @classmethod
    def apply(
        cls,
        latent,
        speech_start_s: float,
        speech_end_s: float,
        fps: float = 24.0,
        head_margin_s: float = 0.2,
        tail_margin_s: float = 0.4,
        ramp_steps: int = 2,
        speech_spans: str = "",
        freeze_video: bool = False,
    ):
        video, audio = _split_streams(latent, "Audio Window Lock expects a MiniMax H3 AV latent")
        mask = latent.get("noise_mask") if isinstance(latent, Mapping) else None
        if mask is None:
            raise ValueError(
                "Audio Window Lock expects a locked digital_human latent (noise_mask "
                "with an audio stream). An unlocked latent needs no window lock."
            )
        video_mask, audio_mask = mask.tensors[0], mask.tensors[1]
        if not isinstance(audio_mask, torch.Tensor) or audio_mask.ndim != 4:
            raise ValueError("Audio Window Lock could not read the packed audio mask")

        frames = _frame_count(int(video.shape[2]))
        duration_s = frames / float(fps)
        steps = int(audio.shape[-1])
        steps_per_second = steps / duration_s

        # Resolve the locked speech spans (seconds, pre-margin). Multi-span mode
        # (JSON list of [start, end] pairs) replaces the single-window form; a
        # single window is just a one-span list.
        raw_spans: list[tuple[float, float]] = []
        if str(speech_spans or "").strip():
            try:
                parsed = json.loads(str(speech_spans))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Audio Window Lock: speech_spans is not valid JSON: {exc}") from None
            if not isinstance(parsed, list) or not parsed:
                raise ValueError("Audio Window Lock: speech_spans must be a non-empty JSON list of [start, end] pairs")
            for pair in parsed:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(f"Audio Window Lock: speech_spans entry is not a [start, end] pair: {pair!r}")
                s, e = float(pair[0]), float(pair[1])
                if e <= s:
                    raise ValueError(f"Audio Window Lock: speech span end <= start: {pair!r}")
                raw_spans.append((s, e))
        else:
            raw_spans = [(float(speech_start_s), float(speech_end_s))]

        # Expand by margins, clamp to the clip, then merge overlapping/touching
        # spans (adjacent lines whose margins bridge the gap become one span).
        ramp = max(0, int(ramp_steps))
        expanded = []
        for s, e in raw_spans:
            lo = int(min(steps, max(0, round((s - float(head_margin_s)) * steps_per_second))))
            hi = int(min(steps, max(0, round((e + float(tail_margin_s)) * steps_per_second))))
            expanded.append([lo, hi])
        expanded.sort(key=lambda item: item[0])
        merged: list[list[int]] = []
        for span in expanded:
            if merged and span[0] <= merged[-1][1] + ramp:
                # overlapping or ramp-bridging: extend the previous span
                merged[-1][1] = max(merged[-1][1], span[1])
            else:
                merged.append([span[0], span[1]])
        if merged and merged[0][0] <= ramp:
            merged[0][0] = 0
        for span in merged:
            if span[1] >= steps - ramp:
                span[1] = steps

        new_audio_mask = torch.ones_like(audio_mask)
        for lo, hi in merged:
            if hi > lo:
                new_audio_mask[..., lo:hi] = 0.0
            # linear ramps outward from both window edges: 0 at the edge -> 1 outside
            head_start = max(0, lo - ramp)
            if head_start < lo:
                span = lo - head_start
                weights = torch.linspace(1.0, 0.0, span + 1, device=audio_mask.device, dtype=audio_mask.dtype)[1:]
                new_audio_mask[..., head_start:lo] = weights.view(*([1] * (audio_mask.ndim - 1)), span)
            tail_end = min(steps, hi + ramp)
            if hi < tail_end:
                span = tail_end - hi
                weights = torch.linspace(0.0, 1.0, span + 1, device=audio_mask.device, dtype=audio_mask.dtype)[1:]
                new_audio_mask[..., hi:tail_end] = weights.view(*([1] * (audio_mask.ndim - 1)), span)

        # Zero the reference audio latent outside the locked core window: Easy's
        # _fit_audio_latent repeats the track's last latent frame across the
        # tail, which leaks quasi-speech content into the generated region.
        new_audio = audio.clone()
        cursor = 0
        for lo, hi in merged:
            if lo > cursor:
                new_audio[..., cursor:lo] = 0.0
            cursor = max(cursor, hi)
        if cursor < steps:
            new_audio[..., cursor:] = 0.0

        # Two-pass mode: pin the entire video stream so it cannot change while
        # the audio tail regenerates against the finished picture.
        if freeze_video:
            video_mask = torch.zeros_like(video_mask)

        packed = _pack_latent_masks(video, new_audio, video_mask, new_audio_mask)
        locked_s = sum(hi - lo for lo, hi in merged) / steps_per_second
        span_text = ", ".join(f"{lo / steps_per_second:.2f}-{hi / steps_per_second:.2f}s" for lo, hi in merged)
        print(
            f"[H3 LatentChain] Audio Window Lock: {len(merged)} locked span(s) [{span_text}] "
            f"of {duration_s:.2f}s (steps {steps}, margins +{float(head_margin_s):.2f}/{float(tail_margin_s):.2f}s, "
            f"locked total {locked_s:.2f}s); "
            f"audio generation UNLOCKED outside the spans, ramp {ramp} steps, "
            f"reference audio zeroed outside the spans"
            + (", video FROZEN (two-pass audio refinement)" if freeze_video else "")
        )
        return (packed,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ChainSave": MiniMaxH3ChainSave,
    "MiniMaxH3ChainLoad": MiniMaxH3ChainLoad,
    "MiniMaxH3ChainLoadFromFile": MiniMaxH3ChainLoadFromFile,
    "MiniMaxH3TakeGuide": MiniMaxH3TakeGuide,
    "MiniMaxH3KeyframeRescale": MiniMaxH3KeyframeRescale,
    "MiniMaxH3AudioWindowLock": MiniMaxH3AudioWindowLock,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ChainSave": "MiniMax H3 Chain Save",
    "MiniMaxH3ChainLoad": "MiniMax H3 Chain Load",
    "MiniMaxH3ChainLoadFromFile": "MiniMax H3 Chain Load From File",
    "MiniMaxH3TakeGuide": "MiniMax H3 Take Guide (Seam)",
    "MiniMaxH3KeyframeRescale": "MiniMax H3 Keyframe Rescale (Stage 2)",
    "MiniMaxH3AudioWindowLock": "MiniMax H3 Audio Window Lock",
}
