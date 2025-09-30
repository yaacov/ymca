"""Tests for chat manager functionality."""

import unittest
from unittest.mock import Mock, patch

from modules.chat.chat_manager import ChatManager
from modules.llm.llm import LLM


class TestChatManager(unittest.TestCase):
    """Test cases for ChatManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the LLM and its dependencies
        self.mock_llm = Mock(spec=LLM)
        self.mock_tokenizer = Mock()
        self.mock_llm.tokenizer = self.mock_tokenizer

        # Create chat manager with test configuration
        self.chat_manager = ChatManager(llm=self.mock_llm, history_window=3)
        self.test_system_prompt = "You are a test assistant."

    def test_initialization(self):
        """Test chat manager initialization."""
        self.assertEqual(self.chat_manager.history_window, 3)
        self.assertEqual(len(self.chat_manager.conversation_history), 0)
        self.assertIsNotNone(self.chat_manager.llm)
        self.assertIsNotNone(self.chat_manager.logger)

    def test_system_prompt_in_send_message(self):
        """Test passing system prompt to send_message method."""
        # Mock LLM response
        self.mock_llm.generate_response.return_value = "Hello! How can I help with coding?"
        self.mock_tokenizer.apply_chat_template.return_value = "Formatted prompt with system"

        system_prompt = "You are a coding assistant."
        response = self.chat_manager.send_message("Hello", system_prompt=system_prompt)

        # Check that LLM was called
        self.mock_llm.generate_response.assert_called_once_with("Formatted prompt with system")
        self.assertEqual(response, "Hello! How can I help with coding?")

    def test_history_window_enforcement(self):
        """Test that history window limits are enforced after conversations."""
        # Mock the LLM to return predictable responses
        self.mock_llm.generate_response.return_value = "Response"

        # Have many conversations beyond the window size
        for i in range(10):
            self.chat_manager.send_message(f"Message {i}")

        # Should only keep last 6 messages (3 pairs * 2)
        self.assertEqual(len(self.chat_manager.conversation_history), 6)
        self.assertEqual(self.chat_manager.conversation_history[0]["content"], "Message 7")
        self.assertEqual(self.chat_manager.conversation_history[1]["content"], "Response")

    def test_format_messages_with_chat_template(self):
        """Test message formatting with chat template."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ]
        self.mock_tokenizer.apply_chat_template.return_value = "<s>Formatted messages</s>"

        result = self.chat_manager.format_messages_for_llm(messages)

        self.mock_tokenizer.apply_chat_template.assert_called_once()
        self.assertEqual(result, "<s>Formatted messages</s>")

    def test_format_messages_fallback(self):
        """Test message formatting fallback."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Simulate no chat template
        self.mock_llm.tokenizer = None

        result = self.chat_manager.format_messages_for_llm(messages)

        expected = "<|system|>\nSystem prompt\n" "<|user|>\nHello\n" "<|assistant|>\nHi there!\n" "<|assistant|>\n"
        self.assertEqual(result, expected)

    def test_send_message(self):
        """Test sending a message and getting response."""
        # Mock LLM response
        self.mock_llm.generate_response.return_value = "Hello! How can I help you?"
        self.mock_tokenizer.apply_chat_template.return_value = "Formatted prompt"

        response = self.chat_manager.send_message("Hello")

        # Check that LLM was called with formatted prompt
        self.mock_llm.generate_response.assert_called_once_with("Formatted prompt")

        # Check response
        self.assertEqual(response, "Hello! How can I help you?")

        # Check history was updated
        self.assertEqual(len(self.chat_manager.conversation_history), 2)
        self.assertEqual(self.chat_manager.conversation_history[0], {"role": "user", "content": "Hello"})
        self.assertEqual(self.chat_manager.conversation_history[1], {"role": "assistant", "content": "Hello! How can I help you?"})

    def test_clear_history(self):
        """Test clearing conversation history."""
        # Add some history
        self.chat_manager.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        self.chat_manager.clear_history()

        self.assertEqual(len(self.chat_manager.conversation_history), 0)

    def test_get_history_summary(self):
        """Test getting history summary."""
        # Add some history
        self.chat_manager.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Goodbye!"},
        ]

        summary = self.chat_manager.get_history_summary()

        expected = {
            "total_messages": 4,
            "user_messages": 2,
            "assistant_messages": 2,
            "history_window": 3,
        }
        self.assertEqual(summary, expected)

    def test_set_history_window(self):
        """Test setting history window size."""
        self.chat_manager.set_history_window(5)
        self.assertEqual(self.chat_manager.history_window, 5)

    def test_set_history_window_invalid(self):
        """Test setting invalid history window size."""
        with self.assertRaises(ValueError):
            self.chat_manager.set_history_window(0)

    def test_no_system_prompt(self):
        """Test chat manager without system prompt."""
        chat_manager = ChatManager(self.mock_llm)
        # Mock LLM response
        self.mock_llm.generate_response.return_value = "Hello without system prompt!"
        self.mock_tokenizer.apply_chat_template.return_value = "Formatted prompt"

        # Test sending message without system prompt
        response = chat_manager.send_message("Hello")

        # Test that conversation history starts empty and gets populated
        self.assertEqual(response, "Hello without system prompt!")
        self.assertEqual(len(chat_manager.conversation_history), 2)  # user + assistant

    def test_max_history_tokens_parameter(self):
        """Test that max_history_tokens parameter is accepted."""
        chat_manager = ChatManager(self.mock_llm, history_window=5, max_history_tokens=2048)
        self.assertEqual(chat_manager.max_history_tokens, 2048)


class TestChatManagerIntegration(unittest.TestCase):
    """Integration tests for ChatManager with real Config."""

    @patch("modules.llm.llm.LLM")
    def test_chat_manager_with_config(self, mock_llm_class):
        """Test ChatManager integration with Config."""
        # Create a mock config
        mock_config = {
            "CHAT_SYSTEM_PROMPT": "You are a helpful coding assistant.",
            "CHAT_HISTORY_WINDOW": 5,
            "CHAT_MAX_HISTORY_TOKENS": 2048,
        }

        # Mock LLM instance
        mock_llm = Mock(spec=LLM)
        mock_llm.tokenizer = Mock()
        mock_llm_class.return_value = mock_llm

        # Create chat manager with config values (system prompt passed to send_message)
        chat_manager = ChatManager(mock_llm, history_window=mock_config["CHAT_HISTORY_WINDOW"], max_history_tokens=mock_config["CHAT_MAX_HISTORY_TOKENS"])

        # Verify configuration was applied
        self.assertEqual(chat_manager.history_window, 5)
        self.assertEqual(chat_manager.max_history_tokens, 2048)

        # Test that system prompt can be passed to send_message
        mock_llm.generate_response.return_value = "Test response"
        mock_llm.tokenizer.apply_chat_template.return_value = "Formatted"

        response = chat_manager.send_message("Hello", system_prompt=mock_config["CHAT_SYSTEM_PROMPT"])
        self.assertEqual(response, "Test response")


if __name__ == "__main__":
    unittest.main()
