"""Tests for device detection and handler system."""

from unittest.mock import Mock, patch

import pytest
import torch

from modules.llm.devices.cpu import CPUDeviceHandler
from modules.llm.devices.cuda import GPUDeviceHandler
from modules.llm.devices.device_detector import DeviceDetector
from modules.llm.devices.intel import IntelDeviceHandler
from modules.llm.devices.mps import MPSDeviceHandler


@pytest.fixture
def mock_config():
    """Mock config for testing."""
    config = Mock()
    config.__getitem__ = Mock(
        side_effect=lambda key: {
            "MODEL_NAME": "test-model",
            "DEVICE": "auto",
            "MODEL_CACHE_DIR": "test_cache",
            "USE_4BIT_QUANTIZATION": True,
            "CUDA_MEMORY_FRACTION": 0.9,
            "USE_INTEL_QUANTIZATION": True,
            "MPS_MEMORY_OPTIMIZATION": True,
        }.get(key, None)
    )
    config.get = Mock(
        side_effect=lambda key, default=None: {
            "MODEL_NAME": "test-model",
            "DEVICE": "auto",
            "MODEL_CACHE_DIR": "test_cache",
            "USE_4BIT_QUANTIZATION": True,
            "CUDA_MEMORY_FRACTION": 0.9,
            "USE_INTEL_QUANTIZATION": True,
            "MPS_MEMORY_OPTIMIZATION": True,
        }.get(key, default)
    )
    return config


class TestDeviceDetector:
    """Test device detection."""

    @patch("torch.cuda.is_available")
    def test_detect_cuda(self, mock_cuda_available):
        """Test CUDA detection."""
        mock_cuda_available.return_value = True
        assert DeviceDetector.detect_cuda() is True

        mock_cuda_available.return_value = False
        assert DeviceDetector.detect_cuda() is False

    @patch("torch.backends.mps.is_available")
    def test_detect_mps(self, mock_mps_available):
        """Test MPS detection."""
        mock_torch = Mock()
        mock_torch.backends.mps.is_available.return_value = True

        with patch("torch.backends", mock_torch.backends):
            assert DeviceDetector.detect_mps() is True

        with patch("torch.backends", Mock(spec=[])):
            assert DeviceDetector.detect_mps() is False

    @patch("builtins.open", create=True)
    def test_detect_intel_cpu(self, mock_open):
        """Test Intel CPU detection via /proc/cpuinfo."""
        mock_file_content = "vendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Core(TM) i7-9750H"
        mock_file = Mock()
        mock_file.read.return_value = mock_file_content
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=None)

        assert DeviceDetector.detect_intel_cpu() is True

        mock_file_content = "vendor_id\t: AuthenticAMD\nmodel name\t: AMD Ryzen 7 3700X"
        mock_file.read.return_value = mock_file_content
        assert DeviceDetector.detect_intel_cpu() is False

        mock_open.side_effect = FileNotFoundError
        assert DeviceDetector.detect_intel_cpu() is False

        mock_open.side_effect = PermissionError
        assert DeviceDetector.detect_intel_cpu() is False

    @patch("modules.llm.devices.device_detector.DeviceDetector.detect_mps")
    @patch("modules.llm.devices.device_detector.DeviceDetector.detect_cuda")
    @patch("modules.llm.devices.device_detector.DeviceDetector.detect_intel_cpu")
    def test_get_available_devices(self, mock_intel_cpu, mock_cuda, mock_mps):
        """Test getting available devices in priority order."""
        mock_cuda.return_value = True
        mock_mps.return_value = True
        mock_intel_cpu.return_value = True

        devices = DeviceDetector.get_available_devices()
        assert devices == ["cuda", "mps", "intel", "cpu"]

        mock_cuda.return_value = False
        mock_mps.return_value = False
        mock_intel_cpu.return_value = False

        devices = DeviceDetector.get_available_devices()
        assert devices == ["cpu"]

    def test_get_device_handler_class(self):
        """Test getting handler classes."""
        cpu_class = DeviceDetector.get_device_handler_class("cpu")
        assert cpu_class is not None
        assert cpu_class.__name__ == "CPUDeviceHandler"

        cuda_class = DeviceDetector.get_device_handler_class("cuda")
        assert cuda_class is not None
        assert cuda_class.__name__ == "GPUDeviceHandler"

        intel_class = DeviceDetector.get_device_handler_class("intel")
        assert intel_class is not None
        assert intel_class.__name__ == "IntelDeviceHandler"

        mps_class = DeviceDetector.get_device_handler_class("mps")
        assert mps_class is not None
        assert mps_class.__name__ == "MPSDeviceHandler"

        assert DeviceDetector.get_device_handler_class("unknown") is None

    @patch("modules.llm.devices.device_detector.DeviceDetector.get_best_device")
    @patch("modules.llm.devices.device_detector.DeviceDetector.get_device_handler_class")
    def test_create_device_handler(self, mock_get_class, mock_get_best, mock_config):
        """Test creating device handlers."""
        mock_get_best.return_value = "cpu"
        mock_handler_class = Mock()
        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler
        mock_get_class.return_value = mock_handler_class

        handler = DeviceDetector.create_device_handler(mock_config, "auto")
        assert handler is mock_handler
        mock_handler_class.assert_called_once_with(mock_config)

        mock_get_class.reset_mock()
        mock_handler_class.reset_mock()

        handler = DeviceDetector.create_device_handler(mock_config, "cuda")
        assert handler is mock_handler
        mock_get_class.assert_called_once_with("cuda")
        mock_handler_class.assert_called_once_with(mock_config)

        mock_get_class.return_value = None
        with patch(
            "modules.llm.devices.device_detector.DeviceDetector.get_available_devices",
            return_value=["unknown_device", "cpu"],
        ):
            with pytest.raises(ValueError, match="Unknown device type: 'unknown_device'"):
                DeviceDetector.create_device_handler(mock_config, "unknown_device")

        mock_get_class.return_value = mock_handler_class
        with patch(
            "modules.llm.devices.device_detector.DeviceDetector.get_available_devices",
            return_value=["cpu"],
        ):
            with pytest.raises(ValueError, match=r"Requested device 'cuda' is not available\. Available devices:"):
                DeviceDetector.create_device_handler(mock_config, "cuda")


class TestCPUDeviceHandler:
    """Test CPU device handler."""

    def test_cpu_handler_basic(self, mock_config):
        """Test basic CPU handler."""
        handler = CPUDeviceHandler(mock_config)

        assert handler.get_device_name() == "cpu"
        assert handler.get_pipeline_device() == -1

    def test_cpu_model_kwargs(self, mock_config):
        """Test CPU model kwargs."""
        handler = CPUDeviceHandler(mock_config)
        kwargs = handler.get_model_kwargs()

        assert "dtype" in kwargs
        assert kwargs["low_cpu_mem_usage"] is True

    def test_cpu_device_info(self, mock_config):
        """Test CPU device info."""
        with patch("torch.get_num_threads", return_value=8):
            handler = CPUDeviceHandler(mock_config)
            info = handler.get_device_info()

            assert info["device_name"] == "cpu"
            assert info["cpu_count"] == 8
            assert "optimizations" in info


class TestGPUDeviceHandler:
    """Test GPU device handler."""

    @patch("torch.cuda.is_available")
    def test_gpu_handler_basic(self, mock_cuda_available, mock_config):
        """Test basic GPU handler."""
        mock_cuda_available.return_value = True
        handler = GPUDeviceHandler(mock_config)

        assert handler.get_device_name() == "cuda"
        assert handler.get_pipeline_device() == 0

        mock_cuda_available.return_value = False
        assert handler.get_pipeline_device() == -1

    @patch("torch.cuda.is_bf16_supported")
    def test_gpu_optimal_dtype(self, mock_bf16_supported, mock_config):
        """Test GPU optimal dtype selection."""
        handler = GPUDeviceHandler(mock_config)

        mock_bf16_supported.return_value = True
        assert handler._get_optimal_dtype() == torch.bfloat16

        mock_bf16_supported.return_value = False
        assert handler._get_optimal_dtype() == torch.float16


class TestIntelDeviceHandler:
    """Test Intel device handler."""

    def test_intel_handler_basic(self, mock_config):
        """Test basic Intel handler."""
        handler = IntelDeviceHandler(mock_config)

        assert handler.get_device_name() == "cpu"
        assert handler.get_pipeline_device() == -1


class TestMPSDeviceHandler:
    """Test MPS device handler."""

    def test_mps_handler_basic(self, mock_config):
        """Test basic MPS handler."""
        handler = MPSDeviceHandler(mock_config)

        assert handler.get_device_name() == "mps"
        assert handler.get_pipeline_device() == "mps"

    def test_mps_optimal_dtype(self, mock_config):
        """Test MPS optimal dtype."""
        handler = MPSDeviceHandler(mock_config)
        assert handler._get_optimal_dtype() == torch.float16
