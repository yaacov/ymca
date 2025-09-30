"""Tests for the LLM module."""

from unittest.mock import Mock, patch

import pytest

from modules.config.config import Config
from modules.llm.llm import LLM


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = Mock(spec=Config)
    config.__getitem__ = Mock(
        side_effect=lambda key: {
            "MODEL_NAME": "test-model",
            "DEVICE": "auto",
            "MODEL_CACHE_DIR": "test_cache",
            "MAX_LENGTH": 512,
            "TEMPERATURE": 0.7,
            "TOP_P": 0.9,
            "DO_SAMPLE": True,
            "USE_4BIT_QUANTIZATION": True,
            "CUDA_MEMORY_FRACTION": 0.9,
            "USE_GRADIENT_CHECKPOINTING": True,
            "USE_TORCH_COMPILE": True,
            "USE_FLASH_ATTENTION": True,
            "USE_INTEL_QUANTIZATION": True,
            "INTEL_CALIBRATION_SIZE": 100,
            "MPS_MEMORY_OPTIMIZATION": True,
            "MPS_MEMORY_FRACTION": 0.8,
        }.get(key, None)
    )
    config.get = Mock(
        side_effect=lambda key, default=None: {
            "MODEL_NAME": "test-model",
            "DEVICE": "auto",
            "MODEL_CACHE_DIR": "test_cache",
            "MAX_LENGTH": 512,
            "TEMPERATURE": 0.7,
            "TOP_P": 0.9,
            "DO_SAMPLE": True,
            "USE_4BIT_QUANTIZATION": True,
            "CUDA_MEMORY_FRACTION": 0.9,
            "USE_GRADIENT_CHECKPOINTING": True,
            "USE_TORCH_COMPILE": True,
            "USE_FLASH_ATTENTION": True,
            "USE_INTEL_QUANTIZATION": True,
            "INTEL_CALIBRATION_SIZE": 100,
            "MPS_MEMORY_OPTIMIZATION": True,
            "MPS_MEMORY_FRACTION": 0.8,
        }.get(key, default)
    )
    return config


@pytest.fixture
def mock_device_handler():
    """Create a mock device handler for testing."""
    handler = Mock()
    handler.device_name = "cpu"
    handler.model = None
    handler.tokenizer = None
    handler.pipeline = None
    handler.load_model.return_value = True
    handler.is_loaded.return_value = False
    handler.generate_response.return_value = "Test response"
    handler.get_device_info.return_value = {
        "device_name": "cpu",
        "is_available": True,
        "optimizations": ["float32", "low_memory"],
    }
    return handler


@patch("modules.llm.devices.device_detector.DeviceDetector.create_device_handler")
def test_llm_init(mock_create_handler, mock_config, mock_device_handler):
    """Test LLM initialization."""
    mock_create_handler.return_value = mock_device_handler

    llm = LLM(mock_config)

    assert llm.config is mock_config
    assert llm.device_handler is mock_device_handler
    assert llm.current_device == "cpu"


@patch("modules.llm.devices.device_detector.DeviceDetector.create_device_handler")
def test_llm_not_loaded_initially(mock_create_handler, mock_config, mock_device_handler):
    """Test that LLM is not loaded initially."""
    mock_create_handler.return_value = mock_device_handler

    llm = LLM(mock_config)
    assert not llm.is_loaded()


@patch("modules.llm.devices.device_detector.DeviceDetector.create_device_handler")
def test_llm_generate_response_without_model(mock_create_handler, mock_config, mock_device_handler):
    """Test generate_response delegates to device handler."""
    mock_create_handler.return_value = mock_device_handler
    mock_device_handler.generate_response.return_value = "Test response from handler"

    llm = LLM(mock_config)
    response = llm.generate_response("test prompt")

    mock_device_handler.generate_response.assert_called_once_with("test prompt")
    assert response == "Test response from handler"


@patch("modules.llm.devices.device_detector.DeviceDetector.create_device_handler")
def test_llm_no_device_handler(mock_create_handler, mock_config):
    """Test LLM behavior when no device handler is available."""
    mock_create_handler.return_value = None

    llm = LLM(mock_config)

    assert llm.device_handler is None
    assert not llm.is_loaded()
    assert llm.generate_response("test") == "Error: No device handler available"
    assert llm.current_device == "none"


@patch("modules.llm.devices.device_detector.DeviceDetector.create_device_handler")
def test_llm_load_model_success(mock_create_handler, mock_config, mock_device_handler):
    """Test successful model loading through device handler."""
    mock_create_handler.return_value = mock_device_handler
    mock_device_handler.load_model.return_value = True
    mock_device_handler.is_loaded.return_value = True

    llm = LLM(mock_config)
    result = llm.load_model()

    assert result is True
    mock_device_handler.load_model.assert_called_once()
    assert llm.is_loaded()


@patch("modules.llm.devices.device_detector.DeviceDetector.create_device_handler")
def test_llm_load_model_failure(mock_create_handler, mock_config, mock_device_handler):
    """Test model loading failure through device handler."""
    mock_create_handler.return_value = mock_device_handler
    mock_device_handler.load_model.return_value = False
    mock_device_handler.is_loaded.return_value = False

    llm = LLM(mock_config)
    result = llm.load_model()

    assert result is False
    mock_device_handler.load_model.assert_called_once()
    assert not llm.is_loaded()


@patch("modules.llm.devices.device_detector.DeviceDetector.create_device_handler")
def test_llm_get_device_info(mock_create_handler, mock_config, mock_device_handler):
    """Test getting device information."""
    mock_create_handler.return_value = mock_device_handler
    expected_info = {
        "device_name": "cpu",
        "is_available": True,
        "optimizations": ["float32", "low_memory"],
    }
    mock_device_handler.get_device_info.return_value = expected_info

    llm = LLM(mock_config)
    info = llm.get_device_info()

    assert info == expected_info
    mock_device_handler.get_device_info.assert_called_once()


@patch("modules.llm.devices.device_detector.DeviceDetector.get_system_info")
@patch("modules.llm.devices.device_detector.DeviceDetector.create_device_handler")
def test_llm_get_system_info(mock_create_handler, mock_get_system_info, mock_config, mock_device_handler):
    """Test getting system information."""
    mock_create_handler.return_value = mock_device_handler
    expected_info = {
        "platform": "Linux",
        "available_devices": ["gpu", "cpu"],
        "best_device": "gpu",
    }
    mock_get_system_info.return_value = expected_info

    llm = LLM(mock_config)
    info = llm.get_system_info()

    assert info == expected_info
    mock_get_system_info.assert_called_once()


@patch("modules.llm.devices.device_detector.DeviceDetector.create_device_handler")
def test_llm_clear_memory(mock_create_handler, mock_config, mock_device_handler):
    """Test clearing memory cache."""
    mock_create_handler.return_value = mock_device_handler
    mock_device_handler.clear_memory_cache = Mock()

    llm = LLM(mock_config)
    llm.clear_memory()

    mock_device_handler.clear_memory_cache.assert_called_once()


@patch("modules.llm.devices.device_detector.DeviceDetector.create_device_handler")
def test_llm_properties(mock_create_handler, mock_config, mock_device_handler):
    """Test LLM properties (model, tokenizer, pipeline)."""
    mock_create_handler.return_value = mock_device_handler
    mock_device_handler.model = "mock_model"
    mock_device_handler.tokenizer = "mock_tokenizer"
    mock_device_handler.pipeline = "mock_pipeline"

    llm = LLM(mock_config)

    assert llm.model == "mock_model"
    assert llm.tokenizer == "mock_tokenizer"
    assert llm.pipeline == "mock_pipeline"
