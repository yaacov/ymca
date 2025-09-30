"""Integration tests for the application."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from modules.config.config import Config
from modules.llm.devices.base import BaseDeviceHandler
from modules.llm.devices.cpu import CPUDeviceHandler
from modules.llm.devices.cuda import GPUDeviceHandler
from modules.llm.devices.device_detector import DeviceDetector
from modules.llm.devices.intel import IntelDeviceHandler
from modules.llm.devices.mps import MPSDeviceHandler
from modules.llm.llm import LLM


@patch("modules.llm.devices.device_detector.DeviceDetector.create_device_handler")
def test_config_llm_integration(mock_create_handler):
    """Test that Config and LLM work together properly."""
    # Create a temporary config file with device-specific settings
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("MODEL_NAME=test-model\n")
        f.write("DEVICE=cpu\n")
        f.write("MAX_LENGTH=256\n")
        f.write("USE_4BIT_QUANTIZATION=false\n")
        f.write("CUDA_MEMORY_FRACTION=0.8\n")
        temp_path = f.name

    # Mock device handler
    mock_handler = Mock()
    mock_handler.device_name = "cpu"
    mock_handler.is_loaded.return_value = False
    mock_handler.get_device_info.return_value = {"device_name": "cpu"}
    mock_create_handler.return_value = mock_handler

    try:
        # Test that config loads correctly
        config = Config(temp_path)
        assert config["MODEL_NAME"] == "test-model"
        assert config["DEVICE"] == "cpu"
        assert config["MAX_LENGTH"] == 256
        assert config["USE_4BIT_QUANTIZATION"] is False
        assert config["CUDA_MEMORY_FRACTION"] == 0.8

        # Test that LLM can be initialized with the config
        llm = LLM(config)
        assert llm.config is config
        assert llm.current_device == "cpu"
        assert not llm.is_loaded()  # Model not loaded yet

        # Test that device handler was created with config
        mock_create_handler.assert_called_once_with(config, "cpu")

    finally:
        os.unlink(temp_path)


def test_imports_work():
    """Test that all main modules can be imported successfully."""
    # This should not raise any exceptions
    assert Config is not None
    assert LLM is not None
    assert DeviceDetector is not None
    assert BaseDeviceHandler is not None
    assert CPUDeviceHandler is not None
    assert GPUDeviceHandler is not None
    assert IntelDeviceHandler is not None
    assert MPSDeviceHandler is not None


@patch("modules.llm.devices.device_detector.DeviceDetector.create_device_handler")
def test_auto_device_selection(mock_create_handler):
    """Test automatic device selection integration."""
    # Create config with auto device selection
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("DEVICE=auto\n")
        temp_path = f.name

    # Mock device handler
    mock_handler = Mock()
    mock_handler.device_name = "cuda"
    mock_handler.get_device_info.return_value = {
        "device_name": "cuda",
        "optimizations": ["4bit_nf4_quantization", "flash_attention"],
    }
    mock_create_handler.return_value = mock_handler

    try:
        config = Config(temp_path)
        llm = LLM(config)

        # Verify auto selection was used
        mock_create_handler.assert_called_once_with(config, "auto")
        assert llm.current_device == "cuda"

        # Test device info retrieval
        info = llm.get_device_info()
        assert info["device_name"] == "cuda"
        assert "optimizations" in info

    finally:
        os.unlink(temp_path)


@patch("modules.llm.devices.device_detector.DeviceDetector.get_system_info")
@patch("modules.llm.devices.device_detector.DeviceDetector.create_device_handler")
def test_system_info_integration(mock_create_handler, mock_get_system_info):
    """Test system information integration."""
    config = Config("nonexistent.env")

    # Mock system info
    mock_system_info = {
        "platform": "Linux",
        "cuda_available": True,
        "available_devices": ["cuda", "cpu"],
        "best_device": "cuda",
    }
    mock_get_system_info.return_value = mock_system_info

    # Mock device handler
    mock_handler = Mock()
    mock_handler.device_name = "cuda"
    mock_create_handler.return_value = mock_handler

    llm = LLM(config)
    system_info = llm.get_system_info()

    assert system_info == mock_system_info
    mock_get_system_info.assert_called_once()


@patch("modules.llm.devices.device_detector.DeviceDetector.create_device_handler")
def test_device_handler_integration(mock_create_handler):
    """Test device handler creation integration."""
    config = Config("nonexistent.env")

    # Simulate device handler creation for auto device selection
    mock_handler = Mock()
    mock_handler.device_name = "cpu"  # Handler device
    mock_create_handler.return_value = mock_handler

    llm = LLM(config)

    # Verify handler was created correctly
    assert llm.current_device == "cpu"
    assert llm.device_handler is mock_handler


def test_device_error_integration():
    """Test device error behavior integration."""
    # Test that requesting unavailable device raises error
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("DEVICE=cuda\n")  # Request CUDA explicitly
        temp_path = f.name

    try:
        config = Config(temp_path)

        # Mock CUDA as not available
        with patch(
            "modules.llm.devices.device_detector.DeviceDetector.get_available_devices",
            return_value=["cpu"],
        ):
            with pytest.raises(ValueError, match="Requested device 'cuda' is not available"):
                _ = LLM(config)

    finally:
        os.unlink(temp_path)
