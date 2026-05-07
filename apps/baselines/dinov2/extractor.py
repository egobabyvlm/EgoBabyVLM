"""DINOv2 feature extractor with flexible pooling strategies."""

import logging
from enum import Enum
from functools import partial
from pathlib import Path
from typing import ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


CheckpointType = Literal["dinov2", "cvcl_lightning", "cvcl_dino_interleaved"]


class DINOv2Pooling(str, Enum):
    """Pooling strategies for DINOv2 feature extraction."""

    CLS = "cls"
    MEAN_PATCH = "mean_patch"
    CLS_MEAN_PATCH = "cls_mean_patch"
    CONCAT_CLS = "concat_cls"
    CONCAT_CLS_AVGPOOL = "concat_cls_avgpool"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"


class DINOv2FeatureExtractor(nn.Module):
    """DINOv2 feature extractor with flexible pooling strategies.

    Supports:

    - Pretrained hub models (e.g. ``dinov2_vitb14``)
    - Custom DINOv2 checkpoints with config files
    - CLIP/CVCL Lightning checkpoints (``.ckpt``) — extracts the vision encoder backbone
    - Interleaved CLIP-DINO checkpoints (``.pth``) — extracts ``dinov2_teacher`` weights
    """

    HUB_MODELS: ClassVar[list[str]] = [
        "dinov2_vits14",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
        "dinov2_vits14_reg",
        "dinov2_vitb14_reg",
        "dinov2_vitl14_reg",
        "dinov2_vitg14_reg",
    ]

    # Architecture dimensions for different ViT models
    VIT_DIMS: ClassVar[dict[str, int]] = {
        "vits": 384,
        "vit_small": 384,
        "vitb": 768,
        "vit_base": 768,
        "vitl": 1024,
        "vit_large": 1024,
        "vitg": 1536,
        "vit_giant": 1536,
    }

    def __init__(
        self,
        pretrained_weights: Path | str = "dinov2_vitb14",
        config_file: Path | str | None = None,
        checkpoint_key: str = "teacher",
        pooling: str | DINOv2Pooling = DINOv2Pooling.CLS,
        last_n_layers: int = 4,
        *,
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self.pooling = DINOv2Pooling(pooling)
        self.last_n_layers = last_n_layers
        self.normalize = normalize

        self._load_model(pretrained_weights, config_file, checkpoint_key)

    @classmethod
    def _detect_checkpoint_type(cls, checkpoint_path: Path | str) -> CheckpointType:
        """
        Detect the type of checkpoint based on file extension and contents.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            One of: "dinov2", "cvcl_lightning", "cvcl_dino_interleaved"
        """
        checkpoint_str = str(checkpoint_path)

        # Lightning checkpoint
        if checkpoint_str.endswith(".ckpt"):
            return "cvcl_lightning"

        # For .pth files, we need to peek inside
        if checkpoint_str.endswith(".pth"):
            local_path = Path(checkpoint_str)
            checkpoint = torch.load(local_path, map_location="cpu", weights_only=False)

            # Check for interleaved CLIP-DINO checkpoint
            if "dinov2_teacher" in checkpoint:
                return "cvcl_dino_interleaved"

            # Standard DINOv2 checkpoint has 'teacher' or 'model' key
            if "teacher" in checkpoint or "model" in checkpoint:
                return "dinov2"

            # Could also be a raw state dict
            raise ValueError(
                f"Unknown .pth checkpoint format. Expected 'teacher', 'model', or 'dinov2_teacher' key. "
                f"Found keys: {list(checkpoint.keys())[:10]}"
            )

        # Assume it's a hub model or DINOv2 checkpoint if no extension
        return "dinov2"

    @classmethod
    def _infer_architecture_from_dim(cls, embed_dim: int) -> str:
        """Infer ViT architecture string from embedding dimension."""
        for arch, dim in cls.VIT_DIMS.items():
            if dim == embed_dim:
                return arch
        raise ValueError(f"Unknown embedding dimension: {embed_dim}")

    @classmethod
    def _infer_architecture_from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> tuple[str, int]:
        """
        Infer model architecture from state dict.

        Returns:
            Tuple of (architecture_string, patch_size)
        """
        # Try to get embed_dim from common keys
        embed_dim = None

        # Check various possible keys for embedding dimension
        for key in ["cls_token", "pos_embed", "patch_embed.proj.weight"]:
            if key in state_dict:
                if key == "patch_embed.proj.weight":
                    # Shape: (embed_dim, in_channels, patch_h, patch_w)
                    embed_dim = state_dict[key].shape[0]
                    patch_size = state_dict[key].shape[2]
                else:
                    # cls_token: (1, 1, embed_dim), pos_embed: (1, num_patches+1, embed_dim)
                    embed_dim = state_dict[key].shape[-1]
                break

        if embed_dim is None:
            raise ValueError("Could not infer embedding dimension from state dict")

        # Get patch size if not already found
        patch_size = 14  # Default
        if "patch_embed.proj.weight" in state_dict:
            patch_size = state_dict["patch_embed.proj.weight"].shape[2]

        arch = cls._infer_architecture_from_dim(embed_dim)
        return f"{arch}{patch_size}", patch_size

    def _load_model(
        self,
        pretrained_weights: Path | str,
        config_file: Path | str | None,
        checkpoint_key: str,
    ) -> None:
        """Load model from hub, custom checkpoint, or CLIP/CVCL checkpoint."""

        if str(pretrained_weights) in self.HUB_MODELS:
            logger.info("Loading Facebook DINOv2 %s from hub", pretrained_weights)
            self.model = torch.hub.load("facebookresearch/dinov2", str(pretrained_weights))
            self._autocast_dtype = torch.float16
        else:
            # Detect checkpoint type
            checkpoint_type = self._detect_checkpoint_type(pretrained_weights)
            logger.info("Detected checkpoint type: %s", checkpoint_type)

            if checkpoint_type == "cvcl_lightning":
                self._load_from_cvcl_lightning(pretrained_weights)
            elif checkpoint_type == "cvcl_dino_interleaved":
                self._load_from_cvcl_dino_interleaved(pretrained_weights, config_file)
            else:
                self._load_custom_checkpoint(pretrained_weights, config_file, checkpoint_key)

        self._autocast_ctx = partial(torch.amp.autocast, device_type="cuda", enabled=True, dtype=self._autocast_dtype)
        logger.info("Using autocast with dtype: %s", self._autocast_dtype)
        self.model.eval()

    def _load_from_cvcl_lightning(self, checkpoint_path: Path | str) -> None:
        """
        Load DINOv2 backbone from a CLIP/CVCL Lightning checkpoint.

        These checkpoints store the vision encoder at:
        - state_dict["vision_encoder.model.*"] or
        - state_dict["model.image_embed.model.*"]

        The model architecture is inferred from the state dict dimensions.
        """
        local_path = Path(str(checkpoint_path))

        logger.info("Loading CVCL Lightning checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(local_path, map_location="cpu", weights_only=False)

        if "hyper_parameters" not in checkpoint:
            raise ValueError("Lightning checkpoint missing 'hyper_parameters'")

        state_dict = checkpoint.get("state_dict", checkpoint)
        vision_state_dict = self._extract_vision_encoder_state_dict(state_dict)

        vision_state_dict = self._flatten_chunked_blocks(vision_state_dict)

        arch_str, _patch_size = self._infer_architecture_from_state_dict(vision_state_dict)
        logger.info("Inferred vision encoder architecture: %s", arch_str)

        # Build and load model
        self._build_and_load_model(vision_state_dict, arch_str)

    def _load_from_cvcl_dino_interleaved(
        self, checkpoint_path: Path | str, config_file: Path | str | None = None
    ) -> None:
        """
        Load DINOv2 backbone from an interleaved CLIP-DINO checkpoint.

        These checkpoints store the teacher model at checkpoint["dinov2_teacher"].
        The teacher includes the full DINOv2 model (backbone + head).
        """
        local_path = Path(str(checkpoint_path))

        logger.info("Loading interleaved CLIP-DINO checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(local_path, map_location="cpu", weights_only=False)

        if "dinov2_teacher" not in checkpoint:
            raise ValueError("Interleaved checkpoint missing 'dinov2_teacher' key")

        teacher_state_dict = checkpoint["dinov2_teacher"]

        backbone_state_dict = {}
        for key, value in teacher_state_dict.items():
            if key.startswith("backbone."):
                new_key = key.replace("backbone.", "")
                backbone_state_dict[new_key] = value
            elif not key.startswith("dino_head") and not key.startswith("ibot_head"):
                backbone_state_dict[key] = value

        if not backbone_state_dict:
            backbone_state_dict = {
                k: v
                for k, v in teacher_state_dict.items()
                if not k.startswith("dino_head") and not k.startswith("ibot_head")
            }

        backbone_state_dict = self._flatten_chunked_blocks(backbone_state_dict)

        if config_file is not None:
            self._load_custom_checkpoint_with_state_dict(backbone_state_dict, config_file)
        else:
            arch_str, _patch_size = self._infer_architecture_from_state_dict(backbone_state_dict)
            logger.info("Inferred architecture: %s", arch_str)
            self._build_and_load_model(backbone_state_dict, arch_str)

    @staticmethod
    def _flatten_chunked_blocks(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Convert block_chunks > 0 state dict to flat blocks structure.

        When DINOv2 is trained with block_chunks > 0, the blocks are organized as:
            blocks.{chunk_idx}.{local_block_idx}.{param}
        where local_block_idx includes Identity placeholders for proper FSDP sharding.

        The actual global block index equals local_block_idx (the Identity placeholders
        at the start of each chunk don't save any state).

        This function converts to the flat structure expected when block_chunks=0:
            blocks.{global_block_idx}.{param}
        """
        import re

        # Pattern to match chunked blocks: blocks.{chunk}.{block}.{rest}
        chunked_pattern = re.compile(r"^blocks\.(\d+)\.(\d+)\.(.+)$")

        flat_state_dict = {}
        has_chunked = False

        for key, value in state_dict.items():
            match = chunked_pattern.match(key)
            if match:
                has_chunked = True
                _chunk_idx = int(match.group(1))
                local_block_idx = int(match.group(2))
                rest = match.group(3)

                global_block_idx = local_block_idx

                new_key = f"blocks.{global_block_idx}.{rest}"
                flat_state_dict[new_key] = value
                logger.debug("Remapped %s -> %s", key, new_key)
            else:
                flat_state_dict[key] = value

        if has_chunked:
            logger.info("Flattened chunked blocks from checkpoint (block_chunks > 0 -> block_chunks = 0)")

        return flat_state_dict

    def _extract_vision_encoder_state_dict(self, full_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Extract vision encoder backbone weights from a full Lightning state dict.

        Handles multiple naming conventions:
        - vision_encoder.model.* (MultiModalLitModel)
        - model.image_embed.model.* (via MultiModalModel)

        If the extracted weights use torchvision's `VisionTransformer` naming
        (class_token / conv_proj / encoder.layers.encoder_layer_N / self_attention,
        used by the no_pretrain training path), they are remapped to DINOv2's
        naming so the existing DINOv2 ViT loader can build the model.
        """
        vision_state_dict = {}

        # Try different prefixes
        prefixes = [
            "vision_encoder.model.",
            "model.image_embed.model.",
        ]

        for prefix in prefixes:
            for key, value in full_state_dict.items():
                if key.startswith(prefix):
                    # Remove the prefix and also skip the projection head
                    new_key = key[len(prefix) :]
                    # Skip the projection head (e.g., "head.weight", "head.bias")
                    if not new_key.startswith("head"):
                        vision_state_dict[new_key] = value

        if not vision_state_dict:
            # Log available keys for debugging
            sample_keys = list(full_state_dict.keys())[:20]
            raise ValueError(
                f"Could not find vision encoder weights in Lightning checkpoint. Sample keys: {sample_keys}"
            )

        # Detect torchvision-style ViT and remap to DINOv2 naming.
        if "class_token" in vision_state_dict or "conv_proj.weight" in vision_state_dict:
            logger.info("Detected torchvision ViT keys; remapping to DINOv2 naming")
            vision_state_dict = self._remap_torchvision_to_dinov2(vision_state_dict)

        return vision_state_dict

    @staticmethod
    def _remap_torchvision_to_dinov2(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert torchvision.models.vision_transformer.VisionTransformer key names
        to DINOv2 ViT naming. Architectures are structurally identical (same shapes);
        only key strings differ. Final classifier `heads.*` is dropped (not part of backbone)."""
        import re as _re

        out: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            new_key = key
            new_key = new_key.replace("class_token", "cls_token")
            new_key = new_key.replace("conv_proj", "patch_embed.proj")
            new_key = new_key.replace("encoder.pos_embedding", "pos_embed")
            new_key = _re.sub(r"^encoder\.ln\.", "norm.", new_key)

            m = _re.match(r"^encoder\.layers\.encoder_layer_(\d+)\.(.+)$", new_key)
            if m:
                idx, rest = m.group(1), m.group(2)
                rest = rest.replace("ln_1", "norm1").replace("ln_2", "norm2")
                rest = rest.replace("self_attention.in_proj_weight", "attn.qkv.weight")
                rest = rest.replace("self_attention.in_proj_bias", "attn.qkv.bias")
                rest = rest.replace("self_attention.out_proj.weight", "attn.proj.weight")
                rest = rest.replace("self_attention.out_proj.bias", "attn.proj.bias")
                rest = rest.replace("mlp.0.", "mlp.fc1.").replace("mlp.3.", "mlp.fc2.")
                new_key = f"blocks.0.{idx}.{rest}"

            # Drop final classifier head — backbone-only.
            if new_key.startswith("heads"):
                continue
            out[new_key] = value
        return out

    def _build_and_load_model(self, state_dict: dict[str, torch.Tensor], model_arch: str) -> None:
        """
        Build a DINOv2 model from architecture string and load weights.

        Args:
            state_dict: The backbone state dict
            model_arch: Architecture string (e.g., "dinov2_vitb14", "vit_base14", "vitb14")
        """
        # Normalize architecture string
        arch_lower = model_arch.lower()

        # Check if it's a hub model
        if arch_lower in self.HUB_MODELS:
            logger.info("Building model from hub: %s", arch_lower)
            self.model = torch.hub.load("facebookresearch/dinov2", arch_lower)
            self._autocast_dtype = torch.float16
        else:
            self._build_model_from_arch(arch_lower, state_dict=state_dict)

        # Load the state dict
        msg = self.model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded backbone weights with message: %s", msg)

    def _build_model_from_arch(self, arch_str: str, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        """
        Build a DINOv2 ViT model from architecture string.

        Args:
            arch_str: Architecture like "vitb14", "vit_base14", "vit_large14", etc.
            state_dict: Optional state dict to infer FFN type and LayerScale from.
        """
        # Parse architecture
        arch_str = arch_str.replace("dinov2_", "")  # Remove dinov2_ prefix if present

        # Determine model size
        if "vits" in arch_str or "vit_small" in arch_str:
            model_name = "vit_small"
        elif "vitb" in arch_str or "vit_base" in arch_str:
            model_name = "vit_base"
        elif "vitl" in arch_str or "vit_large" in arch_str:
            model_name = "vit_large"
        elif "vitg" in arch_str or "vit_giant" in arch_str:
            model_name = "vit_giant2"
        else:
            raise ValueError(f"Unknown architecture: {arch_str}")

        # Determine patch size
        patch_size = 14  # Default
        if "16" in arch_str:
            patch_size = 16
        elif "14" in arch_str:
            patch_size = 14

        # Infer FFN type and LayerScale from state dict if provided
        ffn_layer = "mlp"  # Default
        init_values = None  # Default: no LayerScale

        if state_dict:
            # Check for SwiGLU FFN: uses w12/w3 instead of fc1/fc2
            if any("mlp.w12" in k for k in state_dict):
                ffn_layer = "swiglufused"
                logger.info("Detected SwiGLU FFN from checkpoint (mlp.w12/w3 keys)")
            # Check for LayerScale: uses ls1.gamma/ls2.gamma
            if any("ls1.gamma" in k for k in state_dict):
                init_values = 1e-5  # Default LayerScale init value
                logger.info("Detected LayerScale from checkpoint (ls1.gamma/ls2.gamma keys)")

        logger.info(
            "Building DINOv2 model: %s with patch_size=%s, ffn_layer=%s, init_values=%s",
            model_name,
            patch_size,
            ffn_layer,
            init_values,
        )

        try:
            from dinov2.models import vision_transformer as dino_vision_transformer

            self.model = dino_vision_transformer.__dict__[model_name](
                patch_size=patch_size,
                num_register_tokens=0,
                block_chunks=0,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            self._autocast_dtype = torch.float16

        except ImportError:
            logger.warning("DINOv2 package not found, falling back to torch.hub")
            hub_name = f"dinov2_{model_name.replace('vit_', 'vit')}{patch_size}"
            if hub_name in self.HUB_MODELS:
                self.model = torch.hub.load("facebookresearch/dinov2", hub_name)
                self._autocast_dtype = torch.float16
            else:
                raise ValueError(f"Cannot build model for architecture: {arch_str}") from None

    def _load_custom_checkpoint_with_state_dict(
        self, state_dict: dict[str, torch.Tensor], config_file: Path | str
    ) -> None:
        """Load model from config and apply state dict."""
        try:
            from dinov2.eval.setup import build_model_from_cfg, get_autocast_dtype
            from omegaconf import OmegaConf
        except ImportError as e:
            raise ImportError(
                "DINOv2 package is required for custom checkpoints. "
                "Install from: https://github.com/facebookresearch/dinov2"
            ) from e

        logger.info("Loading model config from %s", config_file)
        with Path(str(config_file)).open("r") as f:
            config = OmegaConf.load(f)

        self.model, _ = build_model_from_cfg(config, only_teacher=True)
        msg = self.model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded weights with message: %s", msg)

        self._autocast_dtype = get_autocast_dtype(config)

    def _load_custom_checkpoint(
        self,
        pretrained_weights: Path | str,
        config_file: Path | str | None,
        checkpoint_key: str,
    ) -> None:
        """Load model from custom DINOv2 checkpoint with config."""
        try:
            import dinov2.utils.utils as dinov2_utils
            from dinov2.eval.setup import build_model_from_cfg, get_autocast_dtype
            from omegaconf import OmegaConf
        except ImportError as e:
            raise ImportError(
                "DINOv2 package is required for custom checkpoints. "
                "Install from: https://github.com/facebookresearch/dinov2"
            ) from e

        if config_file is None:
            logger.warning("No config file provided, trying to infer from weights path")
            config_file = Path(pretrained_weights).parent.parent.parent / "config.yaml"
            if not Path(str(config_file)).exists():
                raise ValueError(
                    f"Config file not found at inferred location: {config_file}. Please provide a valid config_file."
                )

        logger.info("Loading custom DINO model from %s", pretrained_weights)

        with Path(str(config_file)).open("r") as f:
            config = OmegaConf.load(f)

        self.model, _ = build_model_from_cfg(config, only_teacher=True)

        # Load pretrained weights from a local path.
        pretrained_weights_str = str(pretrained_weights)
        local_weights_path = Path(pretrained_weights_str)
        dinov2_utils.load_pretrained_weights(self.model, local_weights_path, checkpoint_key)

        self._autocast_dtype = get_autocast_dtype(config)

    def _compute_feature_dim(self) -> int:
        """Compute output feature dimension based on pooling strategy."""
        embed_dim = self.model.embed_dim

        match self.pooling:
            case DINOv2Pooling.CLS | DINOv2Pooling.MEAN_PATCH:
                return embed_dim
            case DINOv2Pooling.CLS_MEAN_PATCH:
                return embed_dim * 2
            case DINOv2Pooling.CONCAT_CLS:
                return embed_dim * self.last_n_layers
            case DINOv2Pooling.CONCAT_CLS_AVGPOOL:
                return embed_dim * (self.last_n_layers + 1)
            case DINOv2Pooling.SEMANTIC_SEGMENTATION:
                # For semantic segmentation, concatenate patch tokens from last N layers
                return embed_dim * self.last_n_layers

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the configured pooling strategy.

        Args:
            images: (B, C, H, W) normalized image tensor

        Returns:
            Features of shape (B, D) for most pooling modes, or
            (B, N_patches, D) for semantic_segmentation mode
        """
        with self._autocast_ctx():
            match self.pooling:
                case DINOv2Pooling.CLS:
                    # forward() returns x_norm_clstoken when is_training=False
                    features = self.model(images)

                case DINOv2Pooling.MEAN_PATCH:
                    out = self.model.forward_features(images)
                    features = out["x_norm_patchtokens"].mean(dim=1)

                case DINOv2Pooling.CLS_MEAN_PATCH:
                    out = self.model.forward_features(images)
                    features = torch.cat(
                        [
                            out["x_norm_clstoken"],
                            out["x_norm_patchtokens"].mean(dim=1),
                        ],
                        dim=-1,
                    )

                case DINOv2Pooling.CONCAT_CLS:
                    out = self.model.get_intermediate_layers(images, n=self.last_n_layers, return_class_token=True)
                    features = torch.cat([cls for _, cls in out], dim=-1)

                case DINOv2Pooling.CONCAT_CLS_AVGPOOL:
                    out = self.model.get_intermediate_layers(images, n=self.last_n_layers, return_class_token=True)
                    cls_tokens = torch.cat([cls for _, cls in out], dim=-1)
                    last_patch_avg = out[-1][0].mean(dim=1)
                    features = torch.cat([cls_tokens, last_patch_avg], dim=-1)

                case DINOv2Pooling.SEMANTIC_SEGMENTATION:
                    # Extract patch tokens from last N layers without CLS token
                    # Returns (B, N_patches, D * last_n_layers)
                    out = self.model.get_intermediate_layers(images, n=self.last_n_layers, return_class_token=False)
                    features = torch.cat(out, dim=-1)

        features = features.float()

        if self.normalize:
            features = F.normalize(features, dim=-1)

        return features

    @property
    def feature_dim(self) -> int:
        return self._compute_feature_dim()

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    @property
    def input_size(self) -> tuple[int, int]:
        patch_size = self.model.patch_size
        return (patch_size * 37, patch_size * 37)  # 518 for patch_size=14

    @property
    def normalize_params(self) -> dict[str, list[float]]:
        return {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    def get_intermediate_layers(
        self,
        images: torch.Tensor,
        n: int | list[int] = 4,
        *,
        reshape: bool = True,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> list[torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Get intermediate layer features for dense prediction tasks (depth, segmentation).

        This method is needed for DPT-style heads that require multi-scale features.

        Args:
            images: (B, C, H, W) normalized image tensor
            n: Number of layers to return (from end) or list of specific layer indices
            reshape: Whether to reshape patch tokens to spatial format (B, C, H, W)
            return_class_token: Whether to include CLS token with each layer
            norm: Whether to apply LayerNorm to intermediate outputs.
                  The DINOv2 reference uses norm=False for dense prediction (depth, segmentation).

        Returns:
            If return_class_token=False: List of feature tensors (B, C, H, W) or (B, N, D)
            If return_class_token=True: List of (patch_features, cls_token) tuples
        """
        with self._autocast_ctx():
            features = self.model.get_intermediate_layers(
                images,
                n=n,
                reshape=reshape,
                return_class_token=return_class_token,
                norm=norm,
            )
        if return_class_token:
            return [(p.float(), c.float()) for p, c in features]
        return [f.float() for f in features]

    @property
    def n_blocks(self) -> int:
        """Return the number of transformer blocks in the model."""
        return self.model.n_blocks if hasattr(self.model, "n_blocks") else len(self.model.blocks)

    @property
    def patch_size(self) -> int:
        """Return the patch size of the ViT model."""
        return self.model.patch_size

    def set_pooling_strategy(
        self,
        pooling: str | DINOv2Pooling | None = None,
        last_n_layers: int | None = None,
        *,
        normalize: bool | None = None,
    ) -> "DINOv2FeatureExtractor":
        """
        Configure the pooling strategy for feature extraction.

        This allows reconfiguring the model after instantiation, which is useful
        when a shared backbone is injected into multiple tasks that need different
        pooling strategies.

        Args:
            pooling: Pooling strategy (cls, mean_patch, cls_mean_patch, concat_cls, etc.)
            last_n_layers: Number of layers to use for multi-layer pooling strategies
            normalize: Whether to L2-normalize output features

        Returns:
            Self, for method chaining
        """
        if pooling is not None:
            self.pooling = DINOv2Pooling(pooling)
        if last_n_layers is not None:
            self.last_n_layers = last_n_layers
        if normalize is not None:
            self.normalize = normalize
        return self
