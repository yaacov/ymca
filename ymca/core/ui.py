"""
UI utilities for terminal-based interactions.
"""

import sys
import threading
import time
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# Global console instance for consistent rendering
console = Console()


def print_user_input(text: str):
    """
    Print user input with distinctive styling.
    
    Args:
        text: User's input text
    """
    console.print()  # Add spacing
    user_text = Text()
    user_text.append("🧑 You: ", style="bold cyan")
    user_text.append(text, style="white")
    console.print(user_text)


def print_assistant_response(text: str):
    """
    Print assistant response with markdown formatting in a panel.
    
    Args:
        text: Assistant's response (markdown)
    """
    console.print()  # Add spacing
    console.print("Assistant:", style="dim yellow")
    console.print("─" * console.width, style="dim")
    
    # Render markdown
    md = Markdown(text)
    console.print(md)
    console.print("─" * console.width, style="dim")


def print_system_message(text: str, style: str = "dim"):
    """
    Print system message (tool calls, status updates).
    
    Args:
        text: System message text
        style: Rich style string (default: dim)
    """
    console.print(f"  Info: {text}", style=style)


def print_tool_call(tool_name: str, status: str = "calling"):
    """
    Print tool call status with distinctive styling.
    
    Args:
        tool_name: Name of the tool being called
        status: Status string (e.g., "calling", "completed", "failed")
    """
    if status == "calling":
        console.print(f"  Calling tool: ", style="dim", end="")
        console.print(tool_name, style="")
    elif status == "completed":
        console.print(f"  Tool completed: {tool_name}", style="dim")
    elif status == "failed":
        console.print(f"  Tool failed: {tool_name}", style="dim")


def print_markdown(text: str):
    """
    Print text as formatted markdown in the terminal.
    
    Args:
        text: Markdown text to render
    """
    md = Markdown(text)
    console.print(md)


def print_plain(text: str):
    """
    Print plain text (fallback for non-markdown content).
    
    Args:
        text: Plain text to print
    """
    console.print(text)


class ThinkingSpinner:
    """Display an animated spinner with timer while the model is thinking."""
    
    def __init__(self, message: str = "🤔 Thinking"):
        """
        Initialize the spinner.
        
        Args:
            message: Message to display alongside the spinner
        """
        self.message = message
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.stop_spinner = False
        self.spinner_thread = None
        self.start_time = None
    
    def _spin(self):
        """Spinner animation loop with timer."""
        idx = 0
        # ANSI codes: dim text
        dim = "\033[2m"
        reset = "\033[0m"
        
        while not self.stop_spinner:
            elapsed = time.time() - self.start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            
            if minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"
            
            spinner = self.spinner_chars[idx % len(self.spinner_chars)]
            sys.stdout.write(f"\r{dim}{self.message} {spinner} [{time_str}]{reset}  ")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1
        
        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(self.message) + 20) + "\r")
        sys.stdout.flush()
    
    def __enter__(self):
        """Start the spinner."""
        self.start_time = time.time()
        self.stop_spinner = False
        self.spinner_thread = threading.Thread(target=self._spin)
        self.spinner_thread.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the spinner."""
        self.stop_spinner = True
        if self.spinner_thread:
            self.spinner_thread.join()

