#!/usr/bin/env python3
"""Tests for the filesystem manager."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from modules.config.config import Config
from modules.filesystem.filesystem_manager import FilesystemManager
from modules.filesystem.models import DirectoryListing


class TestFilesystemManager(unittest.TestCase):
    """Test cases for FilesystemManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

        # Mock config
        self.mock_config = Mock(spec=Config)
        self.mock_config.get.side_effect = lambda key, default=None: {
            "FILESYSTEM_BASE_DIR": self.temp_dir,
            "FILESYSTEM_MAX_FILE_SIZE_MB": 1,  # 1MB for testing
            "FILESYSTEM_ALLOWED_EXTENSIONS": ".txt,.md,.json",
            "FILESYSTEM_ENABLE_SUBDIRECTORIES": True,
        }.get(key, default)

        # Create filesystem manager with test config
        self.fs_manager = FilesystemManager(self.mock_config)

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test filesystem manager initialization."""
        self.assertEqual(str(self.fs_manager.base_dir), self.temp_dir)
        self.assertEqual(self.fs_manager.max_file_size_mb, 1)
        self.assertEqual(self.fs_manager.allowed_extensions, {".txt", ".md", ".json"})
        self.assertTrue(self.fs_manager.enable_subdirectories)
        self.assertTrue(self.fs_manager.base_dir.exists())

    def test_validate_path_relative(self):
        """Test path validation with relative paths."""
        # Valid relative path
        path = self.fs_manager._validate_path("test.txt")
        expected = Path(self.temp_dir) / "test.txt"
        self.assertEqual(path, expected)

    def test_validate_path_absolute_within_base(self):
        """Test path validation with absolute path within base directory."""
        test_path = os.path.join(self.temp_dir, "test.txt")
        path = self.fs_manager._validate_path(test_path)
        self.assertEqual(path, Path(test_path))

    def test_validate_path_outside_base_directory(self):
        """Test path validation fails for paths outside base directory."""
        with self.assertRaises(ValueError) as context:
            self.fs_manager._validate_path("../outside.txt")
        self.assertIn("outside the allowed base directory", str(context.exception))

    def test_validate_file_extension_allowed(self):
        """Test file extension validation with allowed extensions."""
        self.assertTrue(self.fs_manager._validate_file_extension(Path("test.txt")))
        self.assertTrue(self.fs_manager._validate_file_extension(Path("README.md")))
        self.assertTrue(self.fs_manager._validate_file_extension(Path("data.json")))

    def test_validate_file_extension_not_allowed(self):
        """Test file extension validation with disallowed extensions."""
        self.assertFalse(self.fs_manager._validate_file_extension(Path("test.py")))
        self.assertFalse(self.fs_manager._validate_file_extension(Path("image.jpg")))

    def test_list_files_empty_directory(self):
        """Test listing files in empty directory."""
        listing = self.fs_manager.list_files()

        self.assertIsInstance(listing, DirectoryListing)
        self.assertEqual(listing.path, self.temp_dir)
        self.assertEqual(len(listing.files), 0)
        self.assertEqual(len(listing.directories), 0)
        self.assertEqual(listing.total_size, 0)

    def test_list_files_with_content(self):
        """Test listing files in directory with content."""
        # Create test files
        test_file1 = Path(self.temp_dir) / "test1.txt"
        test_file2 = Path(self.temp_dir) / "test2.md"
        test_dir = Path(self.temp_dir) / "subdir"

        test_file1.write_text("Content 1")
        test_file2.write_text("Content 2")
        test_dir.mkdir()

        listing = self.fs_manager.list_files()

        self.assertEqual(len(listing.files), 2)
        self.assertEqual(len(listing.directories), 1)
        self.assertGreater(listing.total_size, 0)

        # Check file names
        file_names = {f.name for f in listing.files}
        self.assertEqual(file_names, {"test1.txt", "test2.md"})

        # Check directory names
        dir_names = {d.name for d in listing.directories}
        self.assertEqual(dir_names, {"subdir"})

    def test_list_files_nonexistent_directory(self):
        """Test listing files in nonexistent directory."""
        with self.assertRaises(FileNotFoundError):
            self.fs_manager.list_files("nonexistent")

    def test_read_file_success(self):
        """Test reading file successfully."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        content = "Hello, World!"
        test_file.write_text(content)

        result = self.fs_manager.read_file("test.txt")

        self.assertTrue(result.success)
        self.assertEqual(result.operation_type, "read")
        self.assertEqual(result.metadata["content"], content)
        self.assertIn("file_info", result.metadata)

    def test_read_file_nonexistent(self):
        """Test reading nonexistent file."""
        result = self.fs_manager.read_file("nonexistent.txt")

        self.assertFalse(result.success)
        self.assertEqual(result.operation_type, "read")
        self.assertIn("does not exist", result.message)

    def test_read_file_wrong_extension(self):
        """Test reading file with disallowed extension."""
        # Create test file with disallowed extension
        test_file = Path(self.temp_dir) / "test.py"
        test_file.write_text("print('hello')")

        result = self.fs_manager.read_file("test.py")

        self.assertFalse(result.success)
        self.assertIn("extension", result.message)

    def test_read_file_too_large(self):
        """Test reading file that exceeds size limit."""
        # Create large test file (over 1MB limit)
        test_file = Path(self.temp_dir) / "large.txt"
        large_content = "x" * (2 * 1024 * 1024)  # 2MB
        test_file.write_text(large_content)

        result = self.fs_manager.read_file("large.txt")

        self.assertFalse(result.success)
        self.assertIn("exceeds maximum limit", result.message)

    def test_write_file_success(self):
        """Test writing file successfully."""
        content = "Hello, World!"

        result = self.fs_manager.write_file("test.txt", content)

        self.assertTrue(result.success)
        self.assertEqual(result.operation_type, "write")
        self.assertIn("file_info", result.metadata)

        # Verify file was created
        test_file = Path(self.temp_dir) / "test.txt"
        self.assertTrue(test_file.exists())
        self.assertEqual(test_file.read_text(), content)

    def test_write_file_existing_no_overwrite(self):
        """Test writing to existing file without overwrite permission."""
        # Create existing file
        test_file = Path(self.temp_dir) / "existing.txt"
        test_file.write_text("Original content")

        result = self.fs_manager.write_file("existing.txt", "New content", overwrite=False)

        self.assertFalse(result.success)
        self.assertIn("already exists", result.message)

        # Verify original content unchanged
        self.assertEqual(test_file.read_text(), "Original content")

    def test_write_file_existing_with_overwrite(self):
        """Test writing to existing file with overwrite permission."""
        # Create existing file
        test_file = Path(self.temp_dir) / "existing.txt"
        test_file.write_text("Original content")

        new_content = "New content"
        result = self.fs_manager.write_file("existing.txt", new_content, overwrite=True)

        self.assertTrue(result.success)
        self.assertEqual(test_file.read_text(), new_content)

    def test_write_file_wrong_extension(self):
        """Test writing file with disallowed extension."""
        result = self.fs_manager.write_file("test.py", "print('hello')")

        self.assertFalse(result.success)
        self.assertIn("extension", result.message)

    def test_write_file_too_large(self):
        """Test writing content that exceeds size limit."""
        # Try to write content over 1MB limit
        large_content = "x" * (2 * 1024 * 1024)  # 2MB

        result = self.fs_manager.write_file("large.txt", large_content)

        self.assertFalse(result.success)
        self.assertIn("exceeds maximum limit", result.message)

    def test_write_file_with_subdirectory(self):
        """Test writing file in subdirectory."""
        content = "Subdirectory content"

        result = self.fs_manager.write_file("subdir/test.txt", content)

        self.assertTrue(result.success)

        # Verify file was created in subdirectory
        test_file = Path(self.temp_dir) / "subdir" / "test.txt"
        self.assertTrue(test_file.exists())
        self.assertEqual(test_file.read_text(), content)

    def test_delete_file_success(self):
        """Test deleting file successfully."""
        # Create test file
        test_file = Path(self.temp_dir) / "delete_me.txt"
        test_file.write_text("Delete this")

        result = self.fs_manager.delete_file("delete_me.txt")

        self.assertTrue(result.success)
        self.assertEqual(result.operation_type, "delete")
        self.assertFalse(test_file.exists())

    def test_delete_file_nonexistent(self):
        """Test deleting nonexistent file."""
        result = self.fs_manager.delete_file("nonexistent.txt")

        self.assertFalse(result.success)
        self.assertIn("does not exist", result.message)

    def test_create_directory_success(self):
        """Test creating directory successfully."""
        result = self.fs_manager.create_directory("new_dir")

        self.assertTrue(result.success)
        self.assertEqual(result.operation_type, "create_dir")

        # Verify directory was created
        new_dir = Path(self.temp_dir) / "new_dir"
        self.assertTrue(new_dir.exists())
        self.assertTrue(new_dir.is_dir())

    def test_create_directory_existing(self):
        """Test creating existing directory."""
        # Create directory first
        existing_dir = Path(self.temp_dir) / "existing_dir"
        existing_dir.mkdir()

        result = self.fs_manager.create_directory("existing_dir")

        self.assertTrue(result.success)
        self.assertIn("already exists", result.message)

    def test_create_directory_subdirectories_disabled(self):
        """Test creating directory when subdirectories are disabled."""
        # Create filesystem manager with subdirectories disabled
        self.mock_config.get.side_effect = lambda key, default=None: {"FILESYSTEM_BASE_DIR": self.temp_dir, "FILESYSTEM_MAX_FILE_SIZE_MB": 1, "FILESYSTEM_ALLOWED_EXTENSIONS": ".txt,.md,.json", "FILESYSTEM_ENABLE_SUBDIRECTORIES": False}.get(
            key, default
        )

        fs_manager = FilesystemManager(self.mock_config)

        result = fs_manager.create_directory("new_dir")

        self.assertFalse(result.success)
        self.assertIn("not enabled", result.message)

    def test_get_file_info_success(self):
        """Test getting file information successfully."""
        # Create test file
        test_file = Path(self.temp_dir) / "info_test.txt"
        test_file.write_text("Information test")

        result = self.fs_manager.get_file_info("info_test.txt")

        self.assertTrue(result.success)
        self.assertEqual(result.operation_type, "info")
        self.assertIn("file_info", result.metadata)

        file_info = result.metadata["file_info"]
        self.assertEqual(file_info.name, "info_test.txt")
        self.assertFalse(file_info.is_directory)
        self.assertGreater(file_info.size, 0)

    def test_get_file_info_nonexistent(self):
        """Test getting information for nonexistent file."""
        result = self.fs_manager.get_file_info("nonexistent.txt")

        self.assertFalse(result.success)
        self.assertIn("does not exist", result.message)

    def test_parse_allowed_extensions(self):
        """Test parsing allowed extensions from configuration."""
        # Test with spaces
        self.mock_config.get.side_effect = lambda key, default=None: {
            "FILESYSTEM_BASE_DIR": self.temp_dir,
            "FILESYSTEM_MAX_FILE_SIZE_MB": 1,
            "FILESYSTEM_ALLOWED_EXTENSIONS": " .txt , .md , .json ",
            "FILESYSTEM_ENABLE_SUBDIRECTORIES": True,
        }.get(key, default)
        fs_manager = FilesystemManager(self.mock_config)

        expected = {".txt", ".md", ".json"}
        self.assertEqual(fs_manager.allowed_extensions, expected)

    def test_parse_allowed_extensions_empty(self):
        """Test parsing empty allowed extensions."""
        self.mock_config.get.side_effect = lambda key, default=None: {
            "FILESYSTEM_BASE_DIR": self.temp_dir,
            "FILESYSTEM_MAX_FILE_SIZE_MB": 1,
            "FILESYSTEM_ALLOWED_EXTENSIONS": "",
            "FILESYSTEM_ENABLE_SUBDIRECTORIES": True,
        }.get(key, default)
        fs_manager = FilesystemManager(self.mock_config)

        self.assertEqual(fs_manager.allowed_extensions, set())


if __name__ == "__main__":
    unittest.main()
