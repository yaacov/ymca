"""CPU device handler."""

from typing import Any, Dict

import torch

from .base import BaseDeviceHandler


class CPUDeviceHandler(BaseDeviceHandler):
    """CPU device handler."""

    def get_device_name(self) -> str:
        """Return device identifier."""
        return "cpu"

    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get CPU-specific model kwargs."""
        return {
            "dtype": torch.float32,
            "low_cpu_mem_usage": True,
        }

    def get_pipeline_device(self) -> Any:
        """Return device for pipeline (-1 = CPU)."""
        return -1

    def get_torch_dtype(self) -> torch.dtype:
        """Return optimal dtype for CPU."""
        return torch.float32

    def post_process_model(self, model: Any) -> Any:
        """Apply CPU-specific optimizations."""
        model = model.to("cpu")

        try:
            if hasattr(torch.jit, "optimize_for_inference"):
                model = torch.jit.optimize_for_inference(model)
        except Exception:
            pass

        return model

    def get_device_info(self) -> Dict[str, Any]:
        """Get CPU device information."""
        info = super().get_device_info()
        info.update(
            {
                "cpu_count": torch.get_num_threads(),
                "optimizations": ["float32", "low_memory"],
            }
        )
        return info
