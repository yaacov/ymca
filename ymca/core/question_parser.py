"""
Question Parser - Utilities for parsing and cleaning LLM-generated questions.
"""

from typing import List
import logging

logger = logging.getLogger(__name__)


def parse_questions_from_text(text: str, expected_count: int = 3) -> List[str]:
    """
    Parse and clean questions from LLM-generated text.
    
    This handles common LLM output formats:
    - Numbered lists: "1. Question?", "2) Question?"
    - Bulleted lists: "• Question?", "- Question?"
    - Plain text: One question per line
    - Multiple questions on same line: "Question1? Question2?"
    
    Args:
        text: Raw text containing questions
        expected_count: Expected number of questions
        
    Returns:
        List of cleaned question strings
        
    Raises:
        ValueError: If fewer than expected_count questions were found
    """
    questions = []
    
    # Strategy: Split by '?' first to separate questions that might be on same line
    # Then clean and validate each potential question
    parts = text.split('?')
    
    for part in parts:
        part = part.strip()
        
        # Skip empty parts
        if not part:
            continue
        
        # Each part should be a question without the '?'
        # Clean it up and add the '?' back
        cleaned = _remove_question_prefix(part).strip()
        
        if cleaned:
            # Add question mark back
            question = cleaned + '?'
            questions.append(question)
    
    # Limit to expected count
    if len(questions) > expected_count:
        logger.debug(f"Generated {len(questions)} questions, keeping first {expected_count}")
        questions = questions[:expected_count]
    
    # Fail if we don't have enough questions
    if len(questions) < expected_count:
        raise ValueError(
            f"Failed to parse {expected_count} questions from text. "
            f"Only found {len(questions)}. Text: {text[:200]}"
        )
    
    return questions


def _remove_question_prefix(text: str) -> str:
    """
    Remove numbering and bullet prefixes from question text.
    
    Handles formats like:
    - "1. Question?" -> "Question?"
    - "2) Question?" -> "Question?"
    - "• Question?" -> "Question?"
    - "- Question?" -> "Question?"
    
    Args:
        text: Question text with possible prefix
        
    Returns:
        Cleaned question text
    """
    # Strip common prefixes: digits, dots, parens, bullets, dashes, spaces
    return text.lstrip('0123456789.)-• ').strip()

