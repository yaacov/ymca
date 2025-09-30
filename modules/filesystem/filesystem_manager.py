"""Filesystem manager for local file operations with security controls."""

from pathlib import Path
from typing import Any, List, Optional, Set

from ..config.config import Config
from .models import DirectoryListing, FileInfo, FileOperation


class FilesystemManager:
    """Manages local filesystem operations with security and configuration controls."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the filesystem manager with configuration."""
        self.config = config or Config()
        self.base_dir = Path(self.config.get("FILESYSTEM_BASE_DIR", "./data"))
        self.max_file_size_mb: int = self.config.get("FILESYSTEM_MAX_FILE_SIZE_MB", 10)
        self.allowed_extensions = self._parse_allowed_extensions()
        self.enable_subdirectories = self.config.get("FILESYSTEM_ENABLE_SUBDIRECTORIES", True)

        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _parse_allowed_extensions(self) -> Set[str]:
        """Parse allowed extensions from configuration."""
        extensions_str = self.config.get("FILESYSTEM_ALLOWED_EXTENSIONS", ".txt,.md,.json")
        return {ext.strip().lower() for ext in extensions_str.split(",") if ext.strip()}

    def _validate_path(self, file_path: str) -> Path:
        """Validate and normalize a file path within the base directory."""
        # Normalize the path
        path = Path(file_path)

        # If it's not absolute, make it relative to base_dir
        if not path.is_absolute():
            full_path = self.base_dir / path
        else:
            full_path = path

        # Resolve to handle any .. or . components
        try:
            resolved_path = full_path.resolve()
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid path: {e}")

        # Ensure the path is within the base directory
        try:
            resolved_path.relative_to(self.base_dir.resolve())
        except ValueError:
            raise ValueError(f"Path '{file_path}' is outside the allowed base directory")

        return resolved_path

    def _validate_file_extension(self, file_path: Path) -> bool:
        """Check if file extension is allowed."""
        if not self.allowed_extensions:
            return True

        extension = file_path.suffix.lower()
        return extension in self.allowed_extensions

    def _validate_file_size(self, file_path: Path) -> bool:
        """Check if file size is within limits."""
        if not file_path.exists():
            return True

        size_mb = file_path.stat().st_size / (1024 * 1024)
        return size_mb <= self.max_file_size_mb

    def _validate_file_properties(self, file_path: str, operation_type: str, required_checks: List[str], **kwargs: Any) -> Optional[FileOperation]:
        """
        File/directory property validation.

        Args:
            file_path: Path to validate
            operation_type: Type of operation for error reporting
            required_checks: List of checks to perform
            **kwargs: Additional parameters (content_size_mb, allow_overwrite)

        Available checks:
            - "must_exist": Path must exist
            - "must_not_exist": Path must not exist
            - "must_be_file": Path must be a file
            - "must_be_directory": Path must be a directory
            - "check_extension": Validate file extension
            - "check_size": Validate existing file size
            - "check_content_size": Validate content size from kwargs
            - "allow_overwrite": Allow overwriting (default True)

        Returns:
            None if all checks pass, FileOperation error if any check fails
        """
        try:
            # Always validate path security first
            full_path = self._validate_path(file_path)
        except ValueError as e:
            return FileOperation.error_result(operation_type, file_path, str(e))

        # Get file properties once
        exists = full_path.exists()
        is_file = full_path.is_file() if exists else False
        is_dir = full_path.is_dir() if exists else False
        allow_overwrite = kwargs.get("allow_overwrite", True)
        content_size_mb = kwargs.get("content_size_mb")

        # Check each requirement
        for check in required_checks:
            if check == "must_exist" and not exists:
                return FileOperation.error_result(operation_type, file_path, f"Path '{file_path}' does not exist")

            elif check == "must_not_exist" and exists:
                return FileOperation.error_result(operation_type, file_path, f"Path '{file_path}' already exists")

            elif check == "must_be_file" and exists and not is_file:
                return FileOperation.error_result(operation_type, file_path, f"Path '{file_path}' is not a file")

            elif check == "must_be_directory" and exists and not is_dir:
                return FileOperation.error_result(operation_type, file_path, f"Path '{file_path}' is not a directory")

            elif check == "check_extension":
                if not self._validate_file_extension(full_path):
                    return FileOperation.error_result(operation_type, file_path, f"File extension '{full_path.suffix}' is not allowed")

            elif check == "check_size" and exists and is_file:
                if not self._validate_file_size(full_path):
                    return FileOperation.error_result(operation_type, file_path, f"File size exceeds maximum limit of {self.max_file_size_mb}MB")

            elif check == "check_content_size" and content_size_mb is not None:
                if content_size_mb > self.max_file_size_mb:
                    return FileOperation.error_result(operation_type, file_path, f"Content size ({content_size_mb:.2f}MB) exceeds maximum limit of {self.max_file_size_mb}MB")

            elif check == "allow_overwrite" and exists and not allow_overwrite:
                return FileOperation.error_result(operation_type, file_path, f"File '{file_path}' already exists and overwrite=False")

        # All checks passed
        return None

    def list_files(self, directory_path: str = "", include_hidden: bool = False) -> DirectoryListing:
        """
        List files and directories in the specified path.

        Args:
            directory_path: Relative path within base directory (empty for base directory)
            include_hidden: Whether to include hidden files (starting with .)

        Returns:
            DirectoryListing object containing file and directory information
        """
        try:
            if directory_path:
                target_dir = self._validate_path(directory_path)
            else:
                target_dir = self.base_dir

            if not target_dir.exists():
                raise FileNotFoundError(f"Directory '{directory_path}' does not exist")

            if not target_dir.is_dir():
                raise ValueError(f"Path '{directory_path}' is not a directory")

            files = []
            directories = []

            for item in target_dir.iterdir():
                # Skip hidden files unless requested
                if not include_hidden and item.name.startswith("."):
                    continue

                file_info = FileInfo.from_path(item)

                if item.is_file():
                    files.append(file_info)
                elif item.is_dir() and self.enable_subdirectories:
                    directories.append(file_info)

            # Sort by name
            files.sort(key=lambda x: x.name.lower())
            directories.sort(key=lambda x: x.name.lower())

            return DirectoryListing.create_now(str(target_dir), files, directories)

        except FileNotFoundError:
            # Re-raise FileNotFoundError as expected by tests
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to list directory '{directory_path}': {e}")

    def read_file(self, file_path: str, encoding: str = "utf-8") -> FileOperation:
        """
        Read content from a file.

        Args:
            file_path: Path to the file to read
            encoding: Text encoding to use

        Returns:
            FileOperation result with file content or error message
        """
        # Validate file properties
        validation_error = self._validate_file_properties(file_path, "read", ["must_exist", "must_be_file", "check_extension", "check_size"])

        if validation_error:
            return validation_error

        try:
            full_path = self._validate_path(file_path)  # We know this will succeed from validation

            # Read file content
            with open(full_path, "r", encoding=encoding) as f:
                content = f.read()

            file_info = FileInfo.from_path(full_path)

            return FileOperation.success_result("read", file_path, f"Successfully read {len(content)} characters", metadata={"content": content, "file_info": file_info, "encoding": encoding})

        except UnicodeDecodeError as e:
            return FileOperation.error_result("read", file_path, f"Failed to decode file with encoding '{encoding}': {e}")
        except Exception as e:
            return FileOperation.error_result("read", file_path, f"Failed to read file: {e}")

    def write_file(self, file_path: str, content: str, encoding: str = "utf-8", overwrite: bool = False) -> FileOperation:
        """
        Write content to a file.

        Args:
            file_path: Path to the file to write
            content: Content to write to the file
            encoding: Text encoding to use
            overwrite: Whether to overwrite existing files

        Returns:
            FileOperation result with success or error message
        """
        # Calculate content size for validation
        content_size_mb = len(content.encode(encoding)) / (1024 * 1024)

        # Validate file properties
        validation_error = self._validate_file_properties(file_path, "write", ["check_extension", "check_content_size", "allow_overwrite"], content_size_mb=content_size_mb, allow_overwrite=overwrite)

        if validation_error:
            return validation_error

        try:
            full_path = self._validate_path(file_path)  # We know this will succeed from validation

            # Create parent directories if they don't exist
            if not full_path.parent.exists():
                if self.enable_subdirectories:
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    return FileOperation.error_result("write", file_path, "Subdirectories are not enabled")

            # Write file content
            with open(full_path, "w", encoding=encoding) as f:
                f.write(content)

            file_info = FileInfo.from_path(full_path)

            return FileOperation.success_result("write", file_path, f"Successfully wrote {len(content)} characters to file", metadata={"file_info": file_info, "encoding": encoding, "content_length": len(content)})

        except Exception as e:
            return FileOperation.error_result("write", file_path, f"Failed to write file: {e}")

    def delete_file(self, file_path: str) -> FileOperation:
        """
        Delete a file.

        Args:
            file_path: Path to the file to delete

        Returns:
            FileOperation result with success or error message
        """
        # Validate file properties
        validation_error = self._validate_file_properties(file_path, "delete", ["must_exist", "must_be_file"])

        if validation_error:
            return validation_error

        try:
            full_path = self._validate_path(file_path)  # We know this will succeed from validation

            file_info = FileInfo.from_path(full_path)

            # Delete the file
            full_path.unlink()

            return FileOperation.success_result("delete", file_path, "Successfully deleted file", metadata={"deleted_file_info": file_info})

        except Exception as e:
            return FileOperation.error_result("delete", file_path, f"Failed to delete file: {e}")

    def create_directory(self, directory_path: str) -> FileOperation:
        """
        Create a directory.

        Args:
            directory_path: Path to the directory to create

        Returns:
            FileOperation result with success or error message
        """
        if not self.enable_subdirectories:
            return FileOperation.error_result("create_dir", directory_path, "Subdirectories are not enabled")

        # Validate directory path (no specific property checks needed for directory creation)
        validation_error = self._validate_file_properties(directory_path, "create_dir", [])

        if validation_error:
            return validation_error

        try:
            full_path = self._validate_path(directory_path)  # We know this will succeed from validation

            if full_path.exists():
                if full_path.is_dir():
                    return FileOperation.success_result("create_dir", directory_path, "Directory already exists")
                else:
                    return FileOperation.error_result("create_dir", directory_path, f"Path '{directory_path}' exists but is not a directory")

            # Create the directory
            full_path.mkdir(parents=True, exist_ok=True)

            dir_info = FileInfo.from_path(full_path)

            return FileOperation.success_result("create_dir", directory_path, "Successfully created directory", metadata={"directory_info": dir_info})

        except Exception as e:
            return FileOperation.error_result("create_dir", directory_path, f"Failed to create directory: {e}")

    def get_file_info(self, file_path: str) -> FileOperation:
        """
        Get information about a file or directory.

        Args:
            file_path: Path to the file or directory

        Returns:
            FileOperation result with file information or error message
        """
        # Validate file properties
        validation_error = self._validate_file_properties(file_path, "info", ["must_exist"])

        if validation_error:
            return validation_error

        try:
            full_path = self._validate_path(file_path)  # We know this will succeed from validation

            file_info = FileInfo.from_path(full_path)

            return FileOperation.success_result("info", file_path, "Successfully retrieved file information", metadata={"file_info": file_info})

        except Exception as e:
            return FileOperation.error_result("info", file_path, f"Failed to get file information: {e}")
