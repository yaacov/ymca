"""Intel device handler."""

import os
from typing import Any, Dict

import intel_extension_for_pytorch as ipex  # type: ignore
import psutil  # type: ignore
import torch

from .base import BaseDeviceHandler


class IntelDeviceHandler(BaseDeviceHandler):
    """Intel CPU device handler with IPEX optimizations."""

    def get_device_name(self) -> str:
        """Return device identifier."""
        return "cpu"

    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get Intel-optimized model kwargs."""
        return {
            "dtype": self._get_optimal_dtype(),
            "low_cpu_mem_usage": True,
        }

    def get_pipeline_device(self) -> Any:
        """Return device for pipeline."""
        return -1

    def _get_optimal_dtype(self) -> torch.dtype:
        """Get optimal dtype (bfloat16 if supported, else float32)."""
        try:
            torch.tensor([1.0], dtype=torch.bfloat16)
            return torch.bfloat16
        except RuntimeError:
            return torch.float32

    def post_process_model(self, model: Any) -> Any:
        """Apply IPEX optimizations."""
        model = model.to("cpu")
        model = self._apply_ipex_optimizations(model)
        return model

    def _apply_ipex_optimizations(self, model: Any) -> Any:
        """Apply IPEX optimizations."""
        dtype = self._get_optimal_dtype()

        model = ipex.optimize(
            model,
            dtype=dtype,
            level="O1",
            conv_bn_folding=True,
            linear_bn_folding=True,
            weights_prepack=True,
            replace_dropout_with_identity=True,
            optimize_lstm=True,
            split_master_weight_for_bf16=True if dtype == torch.bfloat16 else False,
        )

        self._set_thread_configuration()
        return model

    def _set_thread_configuration(self) -> None:
        """Configure threading for optimal performance."""
        physical_cores = psutil.cpu_count(logical=False)

        if physical_cores:
            torch.set_num_threads(physical_cores)
            os.environ["OMP_NUM_THREADS"] = str(physical_cores)
            os.environ["MKL_NUM_THREADS"] = str(physical_cores)
            os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
            os.environ["KMP_BLOCKTIME"] = "1"

    def _get_quantization_config(self) -> Dict[str, Any]:
        """Get Intel INT8 quantization config."""
        if self.config.get("USE_INTEL_QUANTIZATION", True):
            return {
                "use_intel_int8": True,
                "calibration_dataset_size": self.config.get("INTEL_CALIBRATION_SIZE", 100),
            }
        return {}

    def get_device_info(self) -> Dict[str, Any]:
        """Get Intel device information."""
        info = super().get_device_info()

        bf16_supported = False
        try:
            torch.tensor([1.0], dtype=torch.bfloat16)
            bf16_supported = True
        except RuntimeError:
            pass

        info.update(
            {
                "intel_cpu": True,
                "ipex_available": True,
                "ipex_version": ipex.__version__,
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "bf16_supported": bf16_supported,
                "mkldnn_enabled": torch.backends.mkldnn.enabled,
                "optimization_level": "ipex",
                "optimizations": [
                    "ipex_optimize",
                    "conv_bn_folding",
                    "linear_bn_folding",
                    "weights_prepack",
                    "mkldnn",
                    "thread_optimization",
                ],
            }
        )

        if self.config.get("USE_INTEL_QUANTIZATION", False):
            info["optimizations"].append("int8_quantization")

        return info
