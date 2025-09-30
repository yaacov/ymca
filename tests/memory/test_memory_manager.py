#!/usr/bin/env python3
"""Tests for the memory manager."""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from modules.memory.memory_manager import MemoryManager
from modules.memory.models import Memory, MemoryQuestion


class TestMemoryManager(unittest.TestCase):
    """Test cases for MemoryManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()

        # Mock LLM
        self.mock_llm = Mock()
        self.mock_llm.generate_response.side_effect = ["This is a test summary.", "1. What is this about?\n2. What does this contain?"]  # Summary response  # Questions response

        # Create memory manager with test database
        with patch("modules.memory.memory_manager.EmbeddingService") as mock_embedding_service:
            mock_embedding_instance = Mock()
            mock_embedding_instance.encode_single.return_value = [0.1] * 768  # Mock embedding
            mock_embedding_instance.is_loaded.return_value = True
            mock_embedding_service.return_value = mock_embedding_instance

            self.memory_manager = MemoryManager(llm=self.mock_llm, memory_db_path=self.temp_db.name, embedding_model="test-model", num_questions_per_memory=2)

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary database file
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_store_memory(self):
        """Test storing a memory."""
        content = "This is test content for memory storage."

        # Store memory
        memory_id = self.memory_manager.store_memory(content)

        # Verify memory was stored
        self.assertIsNotNone(memory_id)
        self.assertIsInstance(memory_id, str)

        # Verify LLM was called for summary and questions
        self.assertEqual(self.mock_llm.generate_response.call_count, 2)

        # Retrieve and verify memory
        stored_memory = self.memory_manager.get_memory(memory_id)
        self.assertIsNotNone(stored_memory)
        self.assertEqual(stored_memory.content, content)
        self.assertEqual(stored_memory.summary, "This is a test summary.")
        self.assertEqual(len(stored_memory.questions), 2)

    def test_search_memories_empty(self):
        """Test searching when no memories exist."""
        results = self.memory_manager.search_memories("test query")
        self.assertEqual(len(results), 0)

    def test_get_nonexistent_memory(self):
        """Test getting a memory that doesn't exist."""
        memory = self.memory_manager.get_memory("nonexistent-id")
        self.assertIsNone(memory)

    def test_list_memories_empty(self):
        """Test listing memories when none exist."""
        memories = self.memory_manager.list_memories()
        self.assertEqual(len(memories), 0)

    def test_delete_nonexistent_memory(self):
        """Test deleting a memory that doesn't exist."""
        result = self.memory_manager.delete_memory("nonexistent-id")
        self.assertFalse(result)

    def test_get_stats(self):
        """Test getting memory system statistics."""
        stats = self.memory_manager.get_stats()

        self.assertIsInstance(stats, dict)
        self.assertIn("total_memories", stats)
        self.assertIn("total_questions", stats)
        self.assertIn("questions_per_memory", stats)
        self.assertIn("embedding_service", stats)

        # Should start with 0 memories
        self.assertEqual(stats["total_memories"], 0)
        self.assertEqual(stats["total_questions"], 0)

    def test_no_llm_raises_error(self):
        """Test that memory storage fails when LLM is not available."""
        # Create memory manager without LLM
        with patch("modules.memory.memory_manager.EmbeddingService") as mock_embedding_service:
            mock_embedding_instance = Mock()
            mock_embedding_instance.encode_single.return_value = [0.1] * 768
            mock_embedding_instance.is_loaded.return_value = True
            mock_embedding_service.return_value = mock_embedding_instance

            memory_manager_no_llm = MemoryManager(llm=None, memory_db_path=self.temp_db.name + "_no_llm", embedding_model="test-model", num_questions_per_memory=2)

        # Store memory without LLM should raise error
        content = "This is test content for memory storage without LLM."
        with self.assertRaises(RuntimeError) as context:
            memory_manager_no_llm.store_memory(content)

        self.assertIn("No LLM available", str(context.exception))

        # Clean up
        if os.path.exists(self.temp_db.name + "_no_llm"):
            os.unlink(self.temp_db.name + "_no_llm")


class TestMemoryModels(unittest.TestCase):
    """Test cases for memory data models."""

    def test_memory_question_serialization(self):
        """Test MemoryQuestion serialization and deserialization."""
        question = MemoryQuestion(text="What is this about?", embedding=[0.1, 0.2, 0.3], memory_id="test-memory-id")

        # Test to_dict
        question_dict = question.to_dict()
        self.assertIn("id", question_dict)
        self.assertEqual(question_dict["text"], "What is this about?")
        self.assertEqual(question_dict["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(question_dict["memory_id"], "test-memory-id")

        # Test from_dict
        restored_question = MemoryQuestion.from_dict(question_dict)
        self.assertEqual(restored_question.id, question.id)
        self.assertEqual(restored_question.text, question.text)
        self.assertEqual(restored_question.embedding, question.embedding)
        self.assertEqual(restored_question.memory_id, question.memory_id)

    def test_memory_serialization(self):
        """Test Memory serialization and deserialization."""
        questions = [MemoryQuestion(text="Question 1", embedding=[0.1, 0.2]), MemoryQuestion(text="Question 2", embedding=[0.3, 0.4])]

        memory = Memory(content="Test content", summary="Test summary", questions=questions, tags=["test", "example"])

        # Test to_dict
        memory_dict = memory.to_dict()
        self.assertEqual(memory_dict["content"], "Test content")
        self.assertEqual(memory_dict["summary"], "Test summary")
        self.assertEqual(len(memory_dict["questions"]), 2)
        self.assertEqual(memory_dict["tags"], ["test", "example"])

        # Test from_dict
        restored_memory = Memory.from_dict(memory_dict)
        self.assertEqual(restored_memory.content, memory.content)
        self.assertEqual(restored_memory.summary, memory.summary)
        self.assertEqual(len(restored_memory.questions), 2)
        self.assertEqual(restored_memory.tags, memory.tags)


if __name__ == "__main__":
    unittest.main()
