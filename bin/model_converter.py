#!/usr/bin/env python3
"""
Model Converter - Download and convert Hugging Face models to GGUF format.

This is the main entry point for the model converter tool.
"""

import sys
import logging

try:
    from converter import ModelDownloader, LlamaCppConverter, parse_args, setup_logging
except ImportError as e:
    print(f"Error: Missing required dependencies. Please install requirements.txt")
    print(f"Run: pip install -r requirements.txt")
    sys.exit(1)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the model converter."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    try:
        # Print header
        logger.info("=" * 60)
        logger.info("Model Converter - Hugging Face to GGUF")
        logger.info("=" * 60)
        
        # Initialize downloader
        downloader = ModelDownloader(output_dir=args.output_dir)
        
        # Download model
        model_path = downloader.download(
            model_name=args.model,
            token=args.token,
            force_download=args.force_download
        )
        
        # Get model info
        info = downloader.get_model_info(model_path)
        if info:
            logger.info(f"Model type: {info.get('model_type', 'unknown')}")
            logger.info(f"Architecture: {info.get('architectures', ['unknown'])[0]}")
        
        # Convert to GGUF (unless download-only)
        if not args.download_only:
            logger.info("=" * 60)
            
            converter = LlamaCppConverter()
            gguf_path = converter.convert(
                model_path=model_path,
                quantization=args.quantize,
                output_type=args.output_type
            )
            logger.info(f"✓ GGUF file: {gguf_path}")
        
        # Print footer
        logger.info("=" * 60)
        logger.info("✓ Conversion completed successfully!")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
