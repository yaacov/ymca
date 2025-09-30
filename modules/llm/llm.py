import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from ..config.config import Config
from .devices.base import BaseDeviceHandler
from .devices.device_detector import DeviceDetector


class LLM:
    """Device-agnostic LLM interface."""

    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.device_handler: Optional[BaseDeviceHandler] = None
        self.logger = logger or logging.getLogger("ymca.llm")
        self._prompt_debug_enabled = config.get("DEBUG_SAVE_PROMPTS", False)
        self._prompt_debug_dir = config.get("DEBUG_PROMPTS_DIR", "debug/prompts")
        self._ensure_debug_directory()
        self._initialize_device_handler()

    def _initialize_device_handler(self) -> None:
        """Initialize device handler based on config."""
        device_type = self.config.get("DEVICE", "auto")
        self.device_handler = DeviceDetector.create_device_handler(self.config, device_type)
        if self.device_handler:
            self.logger.debug(f"Initialized device handler: {self.device_handler.device_name}")
        else:
            self.logger.error("Failed to initialize device handler")

    def load_model(self) -> bool:
        """Load model using device handler."""
        if not self.device_handler:
            self.logger.error("Cannot load model: no device handler available")
            return False

        self.logger.info(f"Loading model '{self.config['MODEL_NAME']}' on device '{self.device_handler.device_name}'")
        success = self.device_handler.load_model()

        if success:
            self.logger.info("Model loaded successfully")
        else:
            self.logger.error("Failed to load model")

        return success

    def generate_response(self, formatted_prompt: str, prompt_context: str = "general") -> str:
        """Generate response using pre-formatted prompt."""
        if not self.device_handler:
            return "Error: No device handler available"
        
        # Save prompt for debugging if enabled
        if self._prompt_debug_enabled:
            self._save_prompt_debug(formatted_prompt, prompt_context)
            
        return self.device_handler.generate_response(formatted_prompt)
    
    def generate_response_with_debug(self, formatted_prompt: str, prompt_context: str = "general") -> Tuple[str, Optional[str]]:
        """Generate response and return both response and debug filename (if enabled)."""
        if not self.device_handler:
            return "Error: No device handler available", None
        
        debug_filename = None
        if self._prompt_debug_enabled:
            debug_filename = self._save_prompt_debug(formatted_prompt, prompt_context)
            
        response = self.device_handler.generate_response(formatted_prompt)
        return response, debug_filename

    def _ensure_debug_directory(self) -> None:
        """Ensure debug directory exists."""
        if self._prompt_debug_enabled:
            os.makedirs(self._prompt_debug_dir, exist_ok=True)
            self.logger.debug(f"Debug prompts directory: {self._prompt_debug_dir}")

    def _save_prompt_debug(self, prompt: str, context: str) -> str:
        """Save prompt to debug file and return filename."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
            safe_context = "".join(c for c in context if c.isalnum() or c in "._-")[:30]
            filename = f"prompt_{timestamp}_{safe_context}.txt"
            filepath = os.path.join(self._prompt_debug_dir, filename)
            
            debug_content = f"""=== LLM PROMPT DEBUG ===
Context: {context}
Timestamp: {datetime.now().isoformat()}
Prompt Length: {len(prompt)} characters

=== PROMPT CONTENT ===
{prompt}

=== END PROMPT ===
"""
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(debug_content)
            
            self.logger.debug(f"Saved prompt debug to: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to save prompt debug: {e}")
            return f"debug_save_failed_{timestamp}"

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        if not self.device_handler:
            return False
        return self.device_handler.is_loaded()

    def get_device_info(self) -> Dict[str, Any]:
        """Get current device info."""
        if not self.device_handler:
            return {"error": "No device handler available"}
        return self.device_handler.get_device_info()

    def get_system_info(self) -> Dict[str, Any]:
        """Get system info for device selection."""
        return DeviceDetector.get_system_info()

    def clear_memory(self) -> None:
        """Clear device memory cache."""
        if self.device_handler and hasattr(self.device_handler, "clear_memory_cache"):
            self.device_handler.clear_memory_cache()

    @property
    def current_device(self) -> str:
        """Get current device name."""
        if not self.device_handler:
            return "none"
        return self.device_handler.device_name

    @property
    def model(self) -> Any:
        """Access underlying model."""
        if self.device_handler:
            return self.device_handler.model
        return None

    @property
    def tokenizer(self) -> Any:
        """Access underlying tokenizer."""
        if self.device_handler:
            return self.device_handler.tokenizer
        return None

    @property
    def pipeline(self) -> Any:
        """Access underlying pipeline."""
        if self.device_handler:
            return self.device_handler.pipeline
        return None
