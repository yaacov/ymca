"""Filesystem-related tools for comprehensive file and directory operations."""

from typing import Any, Callable, Dict, List, Tuple, cast


def create_filesystem_tools(filesystem_manager: Any) -> List[Tuple[Dict[str, Any], Callable[..., Any]]]:
    """Create comprehensive filesystem tools for file operations, directory management, and file system exploration."""
    tools = []

    # Read file tool
    read_file_tool = {
        "name": "read_file",
        "description": "Read text content from a file. Use this to examine file contents, read configuration files, or load text data. Supports various text encodings and validates file size limits.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to read (relative to base directory or absolute)"},
                "encoding": {"type": "string", "description": "Text encoding to use", "default": "utf-8"},
            },
            "required": ["file_path"],
        },
        "category": "filesystem",
        "enabled": True,
    }

    def read_file_handler(file_path: str, encoding: str = "utf-8") -> str:
        result = filesystem_manager.read_file(file_path, encoding)
        if result.success:
            content = result.metadata["content"]
            file_info = result.metadata["file_info"]
            return f"File: {file_path}\nSize: {file_info.size} bytes\nModified: {file_info.modified_at}\n\nContent:\n{content}"
        else:
            return f"Error reading file: {result.message}"

    # Write file tool
    write_file_tool = {
        "name": "write_file",
        "description": "Write text content to a file. Use this to create new files or update existing ones with text data. Creates parent directories if needed. Validates file extensions and size limits.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to write (relative to base directory or absolute)"},
                "content": {"type": "string", "description": "Content to write to the file"},
                "encoding": {"type": "string", "description": "Text encoding to use", "default": "utf-8"},
                "overwrite": {"type": "boolean", "description": "Whether to overwrite existing files", "default": False},
            },
            "required": ["file_path", "content"],
        },
        "category": "filesystem",
        "enabled": True,
    }

    def write_file_handler(file_path: str, content: str, encoding: str = "utf-8", overwrite: bool = False) -> str:
        result = filesystem_manager.write_file(file_path, content, encoding, overwrite)
        if result.success:
            file_info = result.metadata["file_info"]
            content_length = result.metadata["content_length"]
            return f"Successfully wrote {content_length} characters to {file_path}\nFile size: {file_info.size} bytes\nCreated: {file_info.created_at}"
        else:
            return f"Error writing file: {result.message}"

    # List files tool
    list_files_tool = {
        "name": "list_files",
        "description": "List files and directories in a specified path. Use this to explore directory contents, see file sizes and dates, or find specific files. Shows both files and subdirectories with detailed information.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory_path": {"type": "string", "description": "Directory path to list (empty string for base directory)", "default": ""},
                "include_hidden": {"type": "boolean", "description": "Whether to include hidden files (starting with .)", "default": False},
            },
            "required": [],
        },
        "category": "filesystem",
        "enabled": True,
    }

    def list_files_handler(directory_path: str = "", include_hidden: bool = False) -> str:
        try:
            listing = filesystem_manager.list_files(directory_path, include_hidden)

            formatted = [f"Directory Listing: {listing.path}"]
            formatted.append(f"Listed at: {listing.created_at}")

            if listing.directories:
                formatted.append(f"\nDirectories ({len(listing.directories)}):")
                for i, dir_info in enumerate(listing.directories, 1):
                    formatted.append(f"{i:2d}. {dir_info.name}/ (modified: {dir_info.modified_at})")

            if listing.files:
                formatted.append(f"\nFiles ({len(listing.files)}):")
                for i, file_info in enumerate(listing.files, 1):
                    size_str = f"{file_info.size:,} bytes"
                    formatted.append(f"{i:2d}. {file_info.name} ({size_str}, modified: {file_info.modified_at})")

            if not listing.files and not listing.directories:
                formatted.append("\nDirectory is empty")

            return "\n".join(formatted)
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    # Register all tools
    tools.extend(
        [
            (read_file_tool, cast(Callable[..., Any], read_file_handler)),
            (write_file_tool, cast(Callable[..., Any], write_file_handler)),
            (list_files_tool, cast(Callable[..., Any], list_files_handler)),
        ]
    )

    return tools
