"""Base class for device handlers."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class BaseDeviceHandler(ABC):
    """Base class for device handlers."""

    def __init__(self, config: Any):
        """Initialize device handler."""
        self.config = config
        self.device_name = self.get_device_name()
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.pipeline: Optional[Any] = None
        self.logger = logging.getLogger(f"ymca.devices.{self.device_name}")

    @abstractmethod
    def get_device_name(self) -> str:
        """Get device name identifier."""
        pass

    @abstractmethod
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get device-specific model kwargs."""
        pass

    @abstractmethod
    def get_pipeline_device(self) -> Any:
        """Get device for pipeline creation."""
        pass

    def get_torch_dtype(self) -> torch.dtype:
        """Get optimal dtype for device."""
        if self.device_name == "cpu":
            return torch.float32
        return torch.float16

    def prepare_model_kwargs(self) -> Dict[str, Any]:
        """Combine base and device-specific model kwargs."""
        base_kwargs = {
            "cache_dir": self.config["MODEL_CACHE_DIR"],
            "trust_remote_code": True,
            "dtype": self.get_torch_dtype(),
            "low_cpu_mem_usage": True,
        }

        device_kwargs = self.get_model_kwargs()
        base_kwargs.update(device_kwargs)

        return base_kwargs

    def load_tokenizer(self) -> bool:
        """Load tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["MODEL_NAME"],
                cache_dir=self.config["MODEL_CACHE_DIR"],
                trust_remote_code=True,
            )

            if self.tokenizer is not None:
                if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None:
                    if hasattr(self.tokenizer, "eos_token"):
                        self.tokenizer.pad_token = self.tokenizer.eos_token

            return True
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            return False

    def load_model(self) -> bool:
        """Load model and create pipeline."""
        try:
            if not self.load_tokenizer():
                self.logger.error("Failed to load tokenizer")
                return False

            model_kwargs = self.prepare_model_kwargs()
            self.logger.debug(f"Model loading kwargs: {model_kwargs}")

            model = AutoModelForCausalLM.from_pretrained(self.config["MODEL_NAME"], **model_kwargs)
            self.model = self.post_process_model(model)

            pipeline_kwargs = {
                "task": "text-generation",
                "model": self.model,
                "tokenizer": self.tokenizer,
            }

            if not self._uses_accelerate_device_map(model_kwargs):
                pipeline_kwargs["device"] = self.get_pipeline_device()

            self.pipeline = pipeline(**pipeline_kwargs)

            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def _uses_accelerate_device_map(self, model_kwargs: Dict[str, Any]) -> bool:
        """Check if using accelerate device mapping."""
        return "device_map" in model_kwargs and model_kwargs["device_map"] is not None

    def post_process_model(self, model: Any) -> Any:
        """Apply device-specific optimizations."""
        return model

    def generate_response(self, formatted_prompt: str) -> str:
        """Generate response using loaded model with pre-formatted prompt."""
        if not self.pipeline or not self.tokenizer:
            return "Error: Model not loaded"

        try:
            params = {
                "max_new_tokens": self.config["MAX_LENGTH"],
                "temperature": self.config["TEMPERATURE"],
                "top_p": self.config["TOP_P"],
                "do_sample": self.config["DO_SAMPLE"],
                "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
                "return_full_text": False,
                "truncation": True,
            }

            self.logger.debug(f"Generation params: max_tokens={params['max_new_tokens']}, " f"temp={params['temperature']}, top_p={params['top_p']}")

            result = self.pipeline(formatted_prompt, **params)
            generated_text = result[0]["generated_text"]
            return generated_text.strip() if generated_text else ""

        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            return f"Error: {str(e)}"

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return bool(self.model and self.tokenizer)

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return {
            "device_name": self.device_name,
            "is_available": True,
            "dtype_info": str(self.get_torch_dtype()),
            "model_loaded": self.is_loaded(),
        }
