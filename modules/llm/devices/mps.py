"""MPS device handler."""

from typing import Any, Dict

import torch

from .base import BaseDeviceHandler


class MPSDeviceHandler(BaseDeviceHandler):
    """MPS (Apple Silicon) device handler."""

    def get_device_name(self) -> str:
        """Return device identifier."""
        return "mps"

    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get MPS-specific model kwargs."""
        return {
            "dtype": self._get_optimal_dtype(),
            "low_cpu_mem_usage": True,
        }

    def get_pipeline_device(self) -> Any:
        """Return device for pipeline."""
        return "mps"

    def _get_optimal_dtype(self) -> torch.dtype:
        """Return optimal dtype for MPS."""
        return torch.float16

    def post_process_model(self, model: Any) -> Any:
        """Apply MPS optimizations with CPU fallback."""
        try:
            model = model.to("mps")

            if hasattr(torch.backends.mps, "enable_fallback"):
                torch.backends.mps.enable_fallback(True)
            torch.mps.empty_cache()

            self._apply_apple_optimizations(model)

        except Exception:
            model = model.to("cpu")

        return model

    def _apply_apple_optimizations(self, model: Any) -> None:
        """Apply Apple Silicon optimizations."""
        try:
            if hasattr(torch.backends.mps, "allow_tf32"):
                torch.backends.mps.allow_tf32 = True

            if self.config.get("MPS_MEMORY_OPTIMIZATION", True):
                if hasattr(torch.mps, "set_per_process_memory_fraction"):
                    memory_fraction = float(self.config.get("MPS_MEMORY_FRACTION", 0.8))
                    torch.mps.set_per_process_memory_fraction(memory_fraction)

        except Exception:
            pass

    def get_device_info(self) -> Dict[str, Any]:
        """Get MPS device information."""
        info = super().get_device_info()

        info.update(
            {
                "mps_available": True,
                "pytorch_version": torch.__version__,
                "metal_available": True,
                "fallback_enabled": getattr(torch.backends.mps, "fallback_enabled", False),
                "optimizations": [
                    "metal_performance_shaders",
                    "float16",
                    "memory_pooling",
                ],
            }
        )

        try:
            if hasattr(torch.mps, "current_allocated_memory"):
                info["memory_allocated"] = torch.mps.current_allocated_memory()
            if hasattr(torch.mps, "driver_allocated_memory"):
                info["memory_driver_allocated"] = torch.mps.driver_allocated_memory()
        except Exception:
            pass

        info["optimizations"].append("scaled_dot_product_attention")

        return info

    def generate_response(self, prompt: str) -> str:
        """Generate response with MPS error handling."""
        try:
            return super().generate_response(prompt)
        except RuntimeError as e:
            if "mps" in str(e).lower():
                torch.mps.empty_cache()
                try:
                    return super().generate_response(prompt)
                except Exception:
                    return f"Error: MPS device error - {str(e)}"
            raise

    def clear_memory_cache(self) -> None:
        """Clear MPS memory cache."""
        torch.mps.empty_cache()
