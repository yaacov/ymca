"""
Statement Parser - Utilities for parsing and cleaning LLM-generated semantic summaries.

Parses declarative statements from LLM output.
"""

from typing import List
import logging
import re

logger = logging.getLogger(__name__)


def parse_questions_from_text(text: str, expected_count: int = 3) -> List[str]:
    """
    Parse and clean semantic summaries from LLM-generated text.
    
    This handles common LLM output formats:
    - Declarative statements: "The tool does X and provides Y capabilities."
    - Numbered lists: "1. Statement", "2) Statement"
    - Bulleted lists: "• Statement", "- Statement"
    - Plain text: One statement per line
    
    Args:
        text: Raw text containing statements
        expected_count: Expected number of statements
        
    Returns:
        List of cleaned statement strings
        
    Raises:
        ValueError: If no statements were found
    """
    # Parse statements line-by-line
    statements = _parse_by_lines(text)
    
    # If we have more than needed, take first N
    if len(statements) > expected_count:
        logger.debug(f"Generated {len(statements)} statements, keeping first {expected_count}")
        statements = statements[:expected_count]
    
    # Validate we got at least one statement
    if len(statements) == 0:
        raise ValueError(
            f"Failed to parse any statements from text. Text: {text[:200]}"
        )
    
    if len(statements) < expected_count:
        logger.warning(
            f"Expected {expected_count} statements but only found {len(statements)}. "
            f"Continuing with {len(statements)} statements."
        )
    
    return statements


def _parse_by_lines(text: str) -> List[str]:
    """
    Parse declarative statements by splitting on newlines.
    
    Args:
        text: Raw text containing statements
        
    Returns:
        List of cleaned statements
    """
    statements = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip lines that look like instructions/prompts
        if _is_instruction_line(line):
            continue
        
        # Clean the line (remove numbering, bullets, trailing punctuation, etc.)
        cleaned = _remove_prefix(line).strip()
        
        # Remove trailing question marks from legacy cached data
        cleaned = cleaned.rstrip('?').strip()
        
        # Validate: must be substantial (at least 10 words)
        if cleaned and len(cleaned.split()) >= 10:
            statements.append(cleaned)
    
    return statements


def _remove_prefix(text: str) -> str:
    """
    Remove numbering and bullet prefixes from text.
    
    Handles formats like:
    - "1. Statement" -> "Statement"
    - "2) Statement" -> "Statement"
    - "• Statement" -> "Statement"
    - "- Statement" -> "Statement"
    - "* Statement" -> "Statement"
    
    Args:
        text: Text with possible prefix
        
    Returns:
        Cleaned text
    """
    # Strip common prefixes: digits, dots, parens, bullets, dashes, asterisks, spaces
    cleaned = text.lstrip('0123456789.)-•*# ').strip()
    
    # Also handle patterns like "1." or "a)" at the start
    cleaned = re.sub(r'^[0-9a-zA-Z]+[\.)]\s+', '', cleaned)
    
    return cleaned


def _is_instruction_line(line: str) -> bool:
    """
    Check if a line looks like an instruction/prompt rather than a statement.
    
    Args:
        line: Line to check
        
    Returns:
        True if line looks like instruction/prompt
    """
    instruction_keywords = [
        'generate',
        'instructions:',
        'text:',
        'semantic summaries:',
        'tool:',
        'description:',
        'output only',
        'each statement',
        'do not number',
        'use present tense'
    ]
    
    line_lower = line.lower()
    return any(keyword in line_lower for keyword in instruction_keywords)

