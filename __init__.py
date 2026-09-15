"""ComfyUI-H3-LatentChain: disk-backed AV take bridging for MiniMax H3 graphs.

A take is a rendered clip's raw video+audio latents stored as one safetensors
file. Save attaches to any stock H3 sampling graph's core LATENT output; Load
plus the Take Guide resume a new clip from a saved take without any RGB
decode/re-encode round trip. The conditioning payload this pack emits
(``minimax_keyframes``) is exactly what comfy_extras/nodes_minimax_h3.py puts
there, so stock samplers and the H3 DiT consume it unchanged.
"""

from .chain import (
    NODE_CLASS_MAPPINGS as _CHAIN_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _CHAIN_DISPLAY,
)

NODE_CLASS_MAPPINGS = dict(_CHAIN_NODES)
NODE_DISPLAY_NAME_MAPPINGS = dict(_CHAIN_DISPLAY)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
