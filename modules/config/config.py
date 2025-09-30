import os
from typing import Any, Dict

from dotenv import dotenv_values


class Config:
    """Configuration manager that loads settings from environment file."""

    def __init__(self, config_path: str = "config.env"):
        self.config: Dict[str, Any] = {}
        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load configuration from environment file and set defaults."""
        # Load environment variables from file if it exists
        env_vars = {}
        if os.path.exists(config_path):
            # Load into a temporary dict to avoid polluting the global environment
            env_vars = dotenv_values(config_path)

        def get_config_value(key: str, default: str) -> str:
            """Get config value from file-specific environment or default."""
            return env_vars.get(key) or os.getenv(key) or default

        self.config = {
            # Basic model configuration
            "MODEL_NAME": get_config_value("MODEL_NAME", "ibm-granite/granite-3.3-8b-instruct"),
            "DEVICE": get_config_value("DEVICE", "auto"),  # auto, cpu, cuda, mps, intel
            "MODEL_CACHE_DIR": get_config_value("MODEL_CACHE_DIR", "models"),
            # Generation parameters
            "MAX_LENGTH": int(get_config_value("MAX_LENGTH", "512")),
            "TEMPERATURE": float(get_config_value("TEMPERATURE", "0.7")),
            "TOP_P": float(get_config_value("TOP_P", "0.9")),
            "DO_SAMPLE": get_config_value("DO_SAMPLE", "true").lower() == "true",
            # CUDA optimizations
            "USE_4BIT_QUANTIZATION": get_config_value("USE_4BIT_QUANTIZATION", "true").lower() == "true",
            "CUDA_MEMORY_FRACTION": float(get_config_value("CUDA_MEMORY_FRACTION", "0.9")),
            "USE_GRADIENT_CHECKPOINTING": get_config_value("USE_GRADIENT_CHECKPOINTING", "true").lower() == "true",
            "USE_TORCH_COMPILE": get_config_value("USE_TORCH_COMPILE", "true").lower() == "true",
            "USE_FLASH_ATTENTION": get_config_value("USE_FLASH_ATTENTION", "true").lower() == "true",
            # Intel optimizations
            "USE_INTEL_QUANTIZATION": get_config_value("USE_INTEL_QUANTIZATION", "true").lower() == "true",
            "INTEL_CALIBRATION_SIZE": int(get_config_value("INTEL_CALIBRATION_SIZE", "100")),
            # MPS (Apple Silicon) optimizations
            "MPS_MEMORY_OPTIMIZATION": get_config_value("MPS_MEMORY_OPTIMIZATION", "true").lower() == "true",
            "MPS_MEMORY_FRACTION": float(get_config_value("MPS_MEMORY_FRACTION", "0.8")),
            # Chat configuration
            "CHAT_SYSTEM_PROMPT": get_config_value("CHAT_SYSTEM_PROMPT", "You are a helpful assistant."),
            "CHAT_HISTORY_WINDOW": int(get_config_value("CHAT_HISTORY_WINDOW", "5")),
            "CHAT_MAX_HISTORY_TOKENS": int(get_config_value("CHAT_MAX_HISTORY_TOKENS", "4096")),
            # Logging configuration
            "LOG_LEVEL": get_config_value("LOG_LEVEL", "INFO").upper(),
            # Debug configuration
            "DEBUG_SAVE_PROMPTS": get_config_value("DEBUG_SAVE_PROMPTS", "false").lower() == "true",
            "DEBUG_PROMPTS_DIR": get_config_value("DEBUG_PROMPTS_DIR", "./debug/prompts"),
            # Web browsing configuration
            "WEB_MAX_REQUESTS_PER_SECOND": float(get_config_value("WEB_MAX_REQUESTS_PER_SECOND", "0.5")),
            "WEB_MAX_REQUESTS_PER_MINUTE": int(get_config_value("WEB_MAX_REQUESTS_PER_MINUTE", "20")),
            "WEB_DEFAULT_SEARCH_ENGINE": get_config_value("WEB_DEFAULT_SEARCH_ENGINE", "duckduckgo"),
            "WEB_MAX_SEARCH_RESULTS": int(get_config_value("WEB_MAX_SEARCH_RESULTS", "10")),
            "WEB_READ_PAGES_BY_DEFAULT": get_config_value("WEB_READ_PAGES_BY_DEFAULT", "true").lower() == "true",
            # Memory configuration
            "MEMORY_DB_PATH": get_config_value("MEMORY_DB_PATH", "memory.db"),
            "MEMORY_EMBEDDING_MODEL": get_config_value("MEMORY_EMBEDDING_MODEL", "ibm-granite/granite-embedding-english-r2"),
            "MEMORY_QUESTIONS_PER_MEMORY": int(get_config_value("MEMORY_QUESTIONS_PER_MEMORY", "2")),
            "MEMORY_MAX_SEARCH_RESULTS": int(get_config_value("MEMORY_MAX_SEARCH_RESULTS", "5")),
            "MEMORY_SIMILARITY_THRESHOLD": float(get_config_value("MEMORY_SIMILARITY_THRESHOLD", "0.1")),
            "MEMORY_MAX_CHUNK_SIZE": int(get_config_value("MEMORY_MAX_CHUNK_SIZE", "1000")),
            "MEMORY_CHUNK_OVERLAP": int(get_config_value("MEMORY_CHUNK_OVERLAP", "200")),
            # Filesystem configuration
            "FILESYSTEM_BASE_DIR": get_config_value("FILESYSTEM_BASE_DIR", "./data"),
            "FILESYSTEM_MAX_FILE_SIZE_MB": int(get_config_value("FILESYSTEM_MAX_FILE_SIZE_MB", "10")),
            "FILESYSTEM_ALLOWED_EXTENSIONS": get_config_value("FILESYSTEM_ALLOWED_EXTENSIONS", ".txt,.md,.json,.csv,.log,.py,.js,.html,.css,.xml,.yaml,.yml"),
            "FILESYSTEM_ENABLE_SUBDIRECTORIES": get_config_value("FILESYSTEM_ENABLE_SUBDIRECTORIES", "true").lower() == "true",
            # Planning and tools configuration
            "PLANNING_ENABLE_WEB_TOOLS": get_config_value("PLANNING_ENABLE_WEB_TOOLS", "true").lower() == "true",
            "PLANNING_ENABLE_FILESYSTEM_TOOLS": get_config_value("PLANNING_ENABLE_FILESYSTEM_TOOLS", "true").lower() == "true",
            "PLANNING_ENABLE_MEMORY_TOOLS": get_config_value("PLANNING_ENABLE_MEMORY_TOOLS", "true").lower() == "true",
            "PLANNING_MAX_STEPS": int(get_config_value("PLANNING_MAX_STEPS", "20")),
            "PLANNING_MAX_ITERATIONS": int(get_config_value("PLANNING_MAX_ITERATIONS", "50")),
            "PLANNING_ENABLE_PARALLEL_EXECUTION": get_config_value("PLANNING_ENABLE_PARALLEL_EXECUTION", "false").lower() == "true",
            "PLANNING_PERSISTENCE_DIR": get_config_value("PLANNING_PERSISTENCE_DIR", "./data/plans"),
            "PLANNING_DEFAULT_MODE": get_config_value("PLANNING_DEFAULT_MODE", "auto"),  # auto, always, never
        }

        os.makedirs(self.config.get("MODEL_CACHE_DIR", "models"), exist_ok=True)
        os.makedirs(self.config.get("FILESYSTEM_BASE_DIR", "./data"), exist_ok=True)
        os.makedirs(self.config.get("PLANNING_PERSISTENCE_DIR", "./data/plans"), exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation."""
        return self.config[key]
