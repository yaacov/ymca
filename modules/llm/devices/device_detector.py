"""Device detection and selection utilities."""

import platform
from typing import Any, Dict, List, Optional, Type

import torch

from .base import BaseDeviceHandler


class DeviceDetector:
    """Device detection and selection."""

    @staticmethod
    def detect_cuda() -> bool:
        """Detect if CUDA is available."""
        return torch.cuda.is_available()

    @staticmethod
    def detect_mps() -> bool:
        """Detect if MPS (Apple Silicon) is available."""
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    @staticmethod
    def detect_intel_cpu() -> bool:
        """Detect if Intel CPU is present."""
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read().lower()
                return "genuineintel" in content or "intel" in content
        except (FileNotFoundError, PermissionError, OSError):
            return False

    @staticmethod
    def get_available_devices() -> List[str]:
        """Get available devices in priority order."""
        devices = []

        if DeviceDetector.detect_cuda():
            devices.append("cuda")

        if DeviceDetector.detect_mps():
            devices.append("mps")

        if DeviceDetector.detect_intel_cpu():
            devices.append("intel")

        devices.append("cpu")

        return devices

    @staticmethod
    def get_best_device(available_devices: List[str]) -> str:
        """Get best available device from priority list."""
        return available_devices[0] if available_devices else "cpu"

    @staticmethod
    def get_device_handler_class(device_type: str) -> Optional[Type[BaseDeviceHandler]]:
        """Get device handler class for given device type."""
        if device_type == "cpu":
            from .cpu import CPUDeviceHandler

            return CPUDeviceHandler
        elif device_type == "cuda":
            from .cuda import GPUDeviceHandler

            return GPUDeviceHandler
        elif device_type == "intel":
            from .intel import IntelDeviceHandler

            return IntelDeviceHandler
        elif device_type == "mps":
            from .mps import MPSDeviceHandler

            return MPSDeviceHandler
        else:
            return None

    @staticmethod
    def create_device_handler(config: Any, device_type: Optional[str] = None) -> BaseDeviceHandler:
        """Create appropriate device handler based on device type and availability."""
        available_devices = DeviceDetector.get_available_devices()

        if device_type is None or device_type == "auto":
            device_type = DeviceDetector.get_best_device(available_devices)

        if device_type not in available_devices:
            raise ValueError(f"Requested device '{device_type}' is not available. Available devices: {available_devices}")

        handler_class = DeviceDetector.get_device_handler_class(device_type)

        if handler_class is None:
            raise ValueError(f"Unknown device type: '{device_type}'")

        return handler_class(config)

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information for device selection."""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": DeviceDetector.detect_cuda(),
            "mps_available": DeviceDetector.detect_mps(),
            "intel_cpu": DeviceDetector.detect_intel_cpu(),
            "available_devices": DeviceDetector.get_available_devices(),
            "best_device": DeviceDetector.get_best_device(DeviceDetector.get_available_devices()),
        }
