"""Tests for the config module."""

import os
import tempfile

from modules.config.config import Config


def test_config_default_values():
    """Test that Config loads default values when no config file exists."""
    config = Config("nonexistent.env")

    # Basic model configuration
    assert config["MODEL_NAME"] == "ibm-granite/granite-3.3-8b-instruct"
    assert config["DEVICE"] == "auto"  # Changed from 'cpu' to 'auto'
    assert config["MODEL_CACHE_DIR"] == "models"

    # Generation parameters
    assert config["MAX_LENGTH"] == 512
    assert config["TEMPERATURE"] == 0.7
    assert config["TOP_P"] == 0.9
    assert config["DO_SAMPLE"] is True

    # CUDA optimizations
    assert config["USE_4BIT_QUANTIZATION"] is True
    assert config["CUDA_MEMORY_FRACTION"] == 0.9
    assert config["USE_GRADIENT_CHECKPOINTING"] is True
    assert config["USE_TORCH_COMPILE"] is True
    assert config["USE_FLASH_ATTENTION"] is True

    # Intel optimizations
    assert config["USE_INTEL_QUANTIZATION"] is True
    assert config["INTEL_CALIBRATION_SIZE"] == 100

    # MPS (Apple Silicon) optimizations
    assert config["MPS_MEMORY_OPTIMIZATION"] is True
    assert config["MPS_MEMORY_FRACTION"] == 0.8

    # Chat configuration
    assert config["CHAT_SYSTEM_PROMPT"] == "You are a helpful assistant."
    assert config["CHAT_HISTORY_WINDOW"] == 5
    assert config["CHAT_MAX_HISTORY_TOKENS"] == 4096


def test_config_custom_values():
    """Test that Config loads values from environment file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("MODEL_NAME=test-model\n")
        f.write("DEVICE=cuda\n")
        f.write("MAX_LENGTH=1024\n")
        f.write("TEMPERATURE=0.5\n")
        f.write("DO_SAMPLE=false\n")
        f.write("USE_4BIT_QUANTIZATION=false\n")
        f.write("CUDA_MEMORY_FRACTION=0.7\n")
        f.write("USE_INTEL_QUANTIZATION=false\n")
        f.write("MPS_MEMORY_FRACTION=0.6\n")
        f.write("CHAT_SYSTEM_PROMPT=You are a coding assistant.\n")
        f.write("CHAT_HISTORY_WINDOW=5\n")
        f.write("CHAT_MAX_HISTORY_TOKENS=2048\n")
        temp_path = f.name

    try:
        config = Config(temp_path)

        assert config["MODEL_NAME"] == "test-model"
        assert config["DEVICE"] == "cuda"
        assert config["MAX_LENGTH"] == 1024
        assert config["TEMPERATURE"] == 0.5
        assert config["DO_SAMPLE"] is False
        assert config["USE_4BIT_QUANTIZATION"] is False
        assert config["CUDA_MEMORY_FRACTION"] == 0.7
        assert config["USE_INTEL_QUANTIZATION"] is False
        assert config["MPS_MEMORY_FRACTION"] == 0.6
        assert config["CHAT_SYSTEM_PROMPT"] == "You are a coding assistant."
        assert config["CHAT_HISTORY_WINDOW"] == 5
        assert config["CHAT_MAX_HISTORY_TOKENS"] == 2048
    finally:
        os.unlink(temp_path)


def test_config_get_method():
    """Test the get method with default values."""
    config = Config("nonexistent.env")

    assert config.get("MODEL_NAME") == "ibm-granite/granite-3.3-8b-instruct"
    assert config.get("NONEXISTENT_KEY") is None
    assert config.get("NONEXISTENT_KEY", "default") == "default"


def test_config_creates_cache_directory():
    """Test that Config creates the model cache directory."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("MODEL_CACHE_DIR=test_cache\n")
        temp_path = f.name

    try:
        _ = Config(temp_path)
        assert os.path.exists("test_cache")
        os.rmdir("test_cache")  # Clean up
    finally:
        os.unlink(temp_path)
