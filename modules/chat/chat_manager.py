"""Chat manager for conversation history and system prompts."""

import logging
from typing import Any, Dict, List, Optional

from ..llm.llm import LLM


class ChatManager:
    """Manages conversation history, system prompts, and LLM interactions."""

    def __init__(
        self,
        llm: LLM,
        history_window: int = 5,
        max_history_tokens: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.llm = llm
        self.history_window = history_window
        self.max_history_tokens = max_history_tokens
        self.conversation_history: List[Dict[str, str]] = []
        self.logger = logger or logging.getLogger("ymca.chat")

    def _trim_history(self) -> None:
        """Keep only the last N conversation pairs in memory."""
        max_messages = self.history_window * 2  # N pairs = N user + N assistant messages
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    def format_messages_for_llm(self, messages: List[Dict[str, str]]) -> str:
        """Format messages using tokenizer's chat template or fallback."""
        if self.llm.tokenizer and hasattr(self.llm.tokenizer, "apply_chat_template"):
            try:
                return str(self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            except Exception:
                pass

        return self._format_messages_fallback(messages)

    def _format_messages_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Fallback formatting when no chat template is available."""
        parts = []

        for msg in messages:
            parts.append(f"<|{msg['role']}|>\n{msg['content']}\n")

        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _build_messages(self, user_input: str, system_prompt: Optional[str] = None, context: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Build complete message list for LLM."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if context:
            messages.extend({"role": "user", "content": ctx} for ctx in context)

        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})

        return messages

    def _store_conversation(self, user_input: str, response: str) -> None:
        """Store conversation pair and trim history."""
        self.conversation_history.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": response}])
        self._trim_history()

    def send_message(self, user_input: str, system_prompt: Optional[str] = None, context: Optional[List[str]] = None) -> str:
        """Send user message and get assistant response."""
        messages = self._build_messages(user_input, system_prompt, context)
        formatted_prompt = self.format_messages_for_llm(messages)

        self.logger.debug(f"Sending formatted prompt to LLM: {formatted_prompt[:200]}{'...' if len(formatted_prompt) > 200 else ''}")

        response = self.llm.generate_response(formatted_prompt)
        self._store_conversation(user_input, response)

        return response

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        self.logger.debug("Conversation history cleared")

    def get_history_summary(self) -> Dict[str, Any]:
        """Get conversation history summary."""
        total_messages = len(self.conversation_history)
        user_messages = len([msg for msg in self.conversation_history if msg["role"] == "user"])
        assistant_messages = len([msg for msg in self.conversation_history if msg["role"] == "assistant"])

        return {
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "history_window": self.history_window,
        }

    def set_history_window(self, window_size: int) -> None:
        """Update history window size."""
        if window_size < 1:
            raise ValueError("History window must be at least 1")

        self.history_window = window_size
