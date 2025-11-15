"""
UI utilities for terminal-based interactions.
"""

import sys
import threading
import time
from rich.console import Console
from rich.markdown import Markdown

# Global console instance for consistent rendering
console = Console()


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
        while not self.stop_spinner:
            elapsed = time.time() - self.start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            
            if minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"
            
            spinner = self.spinner_chars[idx % len(self.spinner_chars)]
            sys.stdout.write(f"\r{self.message} {spinner} [{time_str}]  ")
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

