#!/usr/bin/env python3
"""
Device Information and LLM System Test

This script demonstrates the new device-agnostic LLM system and shows
available devices and their capabilities.
"""

import logging
import sys
import traceback

from modules.chat.chat_manager import ChatManager
from modules.config.config import Config
from modules.llm.devices.device_detector import DeviceDetector
from modules.llm.llm import LLM


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print("=" * 60)


def print_device_info(info: dict, indent: int = 0) -> None:
    """Pretty print device information."""
    prefix = "  " * indent
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_device_info(value, indent + 1)
        elif isinstance(value, list):
            print(f"{prefix}{key}: {', '.join(map(str, value))}")
        else:
            print(f"{prefix}{key}: {value}")


def main() -> int:
    """Main function to test device-agnostic LLM system."""
    print("Device-Agnostic LLM System Test")
    print("Testing device detection and capabilities...")

    try:
        # Initialize configuration
        config = Config()

        # Setup basic logging
        logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
        llm_logger = logging.getLogger("ymca.llm")
        chat_logger = logging.getLogger("ymca.chat")

        print_section("Configuration")
        print(f"Model: {config['MODEL_NAME']}")
        print(f"Device: {config['DEVICE']}")
        print(f"Cache Dir: {config['MODEL_CACHE_DIR']}")

        # Display quantization settings based on device type
        quantization_info = []
        if config["USE_4BIT_QUANTIZATION"]:
            quantization_info.append("4-bit (CUDA)")
        if config["USE_INTEL_QUANTIZATION"]:
            quantization_info.append("Intel")

        print(f"Quantization: {', '.join(quantization_info) if quantization_info else 'Disabled'}")
        print(f"CUDA Memory Fraction: {config['CUDA_MEMORY_FRACTION']}")
        print(f"MPS Memory Fraction: {config['MPS_MEMORY_FRACTION']}")

        # Initialize LLM system
        llm = LLM(config, logger=llm_logger)

        print_section("System Information")
        system_info = llm.get_system_info()
        print_device_info(system_info)

        print_section("Selected Device")
        device_info = llm.get_device_info()
        print_device_info(device_info)

        print(f"\nCurrent device: {llm.current_device}")
        print(f"Model loaded: {llm.is_loaded()}")

        # Test device availability
        print_section("Available Device Types")

        available_devices = DeviceDetector.get_available_devices()
        print(f"Available devices (in priority order): {available_devices}")
        print(f"Best device: {DeviceDetector.get_best_device(available_devices)}")

        # Test only available device handlers
        for device_type in available_devices:
            handler_class = DeviceDetector.get_device_handler_class(device_type)
            if handler_class:
                try:
                    handler = handler_class(config)
                    info = handler.get_device_info()
                    print(f"  {device_type}: Available")
                    optimizations = info.get("optimizations", [])
                    if optimizations:
                        print(f"    Optimizations: {', '.join(optimizations)}")
                except Exception as e:
                    print(f"  {device_type}: Error - {e}")
            else:
                print(f"  {device_type}: Handler not found")

        # Show unavailable devices for completeness
        all_device_types = ["cpu", "cuda", "intel", "mps"]
        unavailable_devices = [dt for dt in all_device_types if dt not in available_devices]
        for device_type in unavailable_devices:
            print(f"  {device_type}: Not available")

        print_section("Model Loading Test")
        print("Attempting to load model...")

        if llm.load_model():
            print("Model loaded successfully!")
            print(f"Device used: {llm.current_device}")

            # Test inference
            print("\nTesting inference...")
            chat_manager = ChatManager(llm, logger=chat_logger)
            test_prompt = "What is machine learning?"
            response = chat_manager.send_message(test_prompt, system_prompt="You are a helpful assistant.")

            print(f"Prompt: {test_prompt}")
            print(f"Response: {response[:200]}...")

        else:
            print("Failed to load model")
            print("This might be due to missing model files or insufficient resources")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

    print_section("Summary")
    print("Device-agnostic LLM system test completed!")
    print("The system automatically detects and uses the best available device.")
    print("You can override device selection by setting DEVICE=cuda|cpu|intel|mps in config.env")

    return 0


if __name__ == "__main__":
    sys.exit(main())
