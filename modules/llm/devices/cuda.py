"""CUDA device handler."""

from typing import Any, Dict, Optional

import torch
from transformers import BitsAndBytesConfig

from .base import BaseDeviceHandler


class GPUDeviceHandler(BaseDeviceHandler):
    """CUDA GPU device handler."""

    def get_device_name(self) -> str:
        """Return device identifier."""
        return "cuda"

    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get CUDA-specific model kwargs."""
        kwargs = {
            "device_map": "auto",
            "dtype": self._get_optimal_dtype(),
        }

        if self._should_use_quantization():
            quant_config = self._get_quantization_config()
            if quant_config:
                kwargs.update(quant_config)

        if self.config.get("CUDA_MEMORY_FRACTION"):
            kwargs["max_memory"] = self._get_memory_config()

        return kwargs

    def get_pipeline_device(self) -> Any:
        """Return device for pipeline (0 for first GPU, -1 if unavailable)."""
        if torch.cuda.is_available():
            return 0
        return -1

    def _get_optimal_dtype(self) -> torch.dtype:
        """Get optimal dtype (bfloat16 if supported, else float16)."""
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _should_use_quantization(self) -> bool:
        """Check if 4-bit quantization should be used."""
        use_quantization = self.config.get("USE_4BIT_QUANTIZATION", True)
        return bool(use_quantization)

    def _get_quantization_config(self) -> Optional[Dict[str, Any]]:
        """Get 4-bit quantization config."""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self._get_optimal_dtype(),
            bnb_4bit_use_double_quant=True,
        )

        return {"quantization_config": quantization_config}

    def _get_memory_config(self) -> Dict[int, str]:
        """Get memory allocation config for GPUs."""
        memory_fraction = float(self.config.get("CUDA_MEMORY_FRACTION", 0.9))

        if torch.cuda.device_count() > 1:
            max_memory = {}
            for i in range(torch.cuda.device_count()):
                total_memory = torch.cuda.get_device_properties(i).total_memory
                max_memory[i] = f"{int(total_memory * memory_fraction / (1024**3))}GB"
            return max_memory
        else:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            return {0: f"{int(total_memory * memory_fraction / (1024**3))}GB"}

    def post_process_model(self, model: Any) -> Any:
        """Apply GPU-specific optimizations."""
        if hasattr(model.config, "use_flash_attention_2"):
            model.config.use_flash_attention_2 = True

        if hasattr(model, "gradient_checkpointing_enable") and self.config.get("USE_GRADIENT_CHECKPOINTING", True):
            model.gradient_checkpointing_enable()

        return model

    def get_device_info(self) -> Dict[str, Any]:
        """Get CUDA device information."""
        info = super().get_device_info()

        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "devices": [],
        }

        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            gpu_info["devices"].append(
                {
                    "id": i,
                    "name": device_props.name,
                    "memory_total": device_props.total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_reserved": torch.cuda.memory_reserved(i),
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                }
            )

        info.update(gpu_info)

        info["quantization_available"] = self._should_use_quantization()
        info["optimizations"] = [
            "float16/bfloat16",
            "device_map_auto",
            "flash_attention",
            "gradient_checkpointing",
            "pipeline_optimization",
        ]

        if self._should_use_quantization():
            info["optimizations"].append("4bit_nf4_quantization")

        return info
