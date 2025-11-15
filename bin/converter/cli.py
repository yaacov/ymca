"""
CLI - Command-line interface for the model converter.
"""

import argparse
import logging

# Default model: IBM Granite 4.0 Hybrid Small
DEFAULT_MODEL = "ibm-granite/granite-4.0-h-small"


def setup_logging(verbose: bool = False):
    """Setup logging configuration.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Download and convert Hugging Face models to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default model (IBM Granite 4.0 Hybrid Small)
  python scripts/model_converter.py
  
  # Download specific model
  python scripts/model_converter.py --model meta-llama/Llama-2-7b-chat-hf
  
  # Download and convert with quantization (K-quant recommended)
  python scripts/model_converter.py --quantize q4_k_m
  
  # Download only (skip conversion)
  python scripts/model_converter.py --download-only
  
  # Use custom output directory
  python scripts/model_converter.py --output-dir ./my_models
  
Available quantization methods:
  K-quants (recommended - better quality):
    - q4_k_m: 4-bit K-quant medium (best for most uses)
    - q4_k_s: 4-bit K-quant small (slightly smaller)
    - q5_k_m: 5-bit K-quant medium (recommended balance)
    - q5_k_s: 5-bit K-quant small
    - q6_k: 6-bit K-quant (very good quality)
    - q3_k_m: 3-bit K-quant medium (aggressive compression)
    - q3_k_s: 3-bit K-quant small
    - q3_k_l: 3-bit K-quant large
    - q2_k: 2-bit K-quant (experimental)
  
  Legacy quants:
    - q4_0: 4-bit legacy
    - q4_1: 4-bit legacy (better quality)
    - q5_0: 5-bit legacy
    - q5_1: 5-bit legacy (better quality)
    - q8_0: 8-bit quantization (high quality)
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Hugging Face model identifier (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory for downloaded models (default: ./models)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (for private models)"
    )
    
    parser.add_argument(
        "--quantize",
        type=str,
        choices=[
            # Legacy quantization methods
            "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
            # K-quant methods (recommended)
            "q2_k", "q3_k_s", "q3_k_m", "q3_k_l",
            "q4_k_s", "q4_k_m",
            "q5_k_s", "q5_k_m",
            "q6_k", "q8_0"
        ],
        default=None,
        help="Quantization method for GGUF conversion (K-quants recommended for better quality)"
    )
    
    parser.add_argument(
        "--output-type",
        type=str,
        choices=["f32", "f16", "q8_0"],
        default="f16",
        help="Output data type for GGUF conversion (default: f16)"
    )
    
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the model, skip GGUF conversion"
    )
    
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if model exists"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

