"""
Model Downloader - Download models from Hugging Face Hub.
"""

import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Handle model downloading from Hugging Face."""
    
    def __init__(self, output_dir: str = "./models"):
        """Initialize the model downloader.
        
        Args:
            output_dir: Directory to save downloaded models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download(
        self,
        model_name: str,
        token: Optional[str] = None,
        force_download: bool = False
    ) -> Path:
        """Download a model from Hugging Face.
        
        Args:
            model_name: Hugging Face model identifier
            token: Hugging Face API token (optional, for private models)
            force_download: Force re-download even if model exists
            
        Returns:
            Path to the downloaded model directory
        """
        logger.info(f"Downloading model: {model_name}")
        
        # Create model-specific directory
        model_dir = self.output_dir / model_name.replace("/", "_")
        safetensor_dir = model_dir / "safetensor"
        safetensor_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info("Downloading model files from Hugging Face...")
            local_dir = snapshot_download(
                repo_id=model_name,
                local_dir=str(safetensor_dir),
                token=token,
                force_download=force_download
            )
            
            logger.info(f"✓ Model downloaded: {local_dir}")
            return Path(local_dir)
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    def get_model_info(self, model_path: Path) -> dict:
        """Get information about a downloaded model.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Dictionary with model information
        """
        try:
            import json
            config_file = model_path / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load model config: {e}")
        
        return {}

