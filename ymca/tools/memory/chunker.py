"""Text chunking utilities with markdown awareness."""

import re
from typing import List


class TextChunker:
    """Handles text chunking with overlap and markdown structure awareness."""
    
    def __init__(self, chunk_size: int = 4000, overlap: int = 400):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def _is_in_code_block(self, text: str, pos: int) -> bool:
        """Check if position is inside a markdown code block."""
        # Count triple backticks before this position
        before_text = text[:pos]
        backtick_count = before_text.count('```')
        
        # If odd number of ``` before this position, we're inside a code block
        return backtick_count % 2 == 1
    
    def _find_code_block_end(self, text: str, pos: int) -> int:
        """Find the end of a code block starting from position."""
        # Find the closing ```
        remaining = text[pos:]
        end_marker = remaining.find('```')
        
        if end_marker >= 0:
            # Move past the closing ``` and any trailing newline
            end_pos = pos + end_marker + 3
            if end_pos < len(text) and text[end_pos] == '\n':
                end_pos += 1
            return end_pos
        
        # If no closing found, return current position
        return pos
    
    def _is_in_table(self, text: str, pos: int) -> bool:
        """Check if position is inside a markdown table."""
        # Look back and forward for table markers
        before = text[max(0, pos-500):pos]
        after = text[pos:min(len(text), pos+500)]
        
        # Count pipes in surrounding context
        before_pipes = before.count('|')
        after_pipes = after.count('|')
        
        # If we see table-like structure (multiple pipes), we're likely in a table
        return before_pipes >= 2 and after_pipes >= 2
    
    def _find_section_boundary(self, text: str, start: int, end: int) -> int:
        """Find a good boundary that respects markdown sections."""
        search_text = text[start:end]
        
        # Priority 1: Try to break at markdown heading (##, ###, etc.)
        for pattern in [r'\n#{1,6}\s', r'\n---+\n', r'\n===+\n']:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # Use the last heading before the end
                last_match = matches[-1]
                boundary = start + last_match.start()
                if boundary > start + self.chunk_size // 2:
                    return boundary
        
        # Priority 2: Try to break at paragraph boundary (double newline)
        last_para = search_text.rfind('\n\n')
        if last_para > self.chunk_size // 2:
            return start + last_para + 2
        
        # Priority 3: Try to break at sentence boundary
        for separator in ['. \n', '.\n', '. ', '! ', '?\n']:
            last_sep = search_text.rfind(separator)
            if last_sep > self.chunk_size // 2:
                return start + last_sep + len(separator)
        
        # Priority 4: Break at any newline
        last_newline = search_text.rfind('\n')
        if last_newline > self.chunk_size // 3:
            return start + last_newline + 1
        
        # Last resort: break at the original end
        return end
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into smaller pieces with overlap, respecting markdown structure.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # If we're starting inside a code block (due to overlap), skip past it
            if start > 0 and self._is_in_code_block(text, start):
                # Find the end of this code block
                code_block_end = self._find_code_block_end(text, start)
                if code_block_end > start:
                    start = code_block_end
                    # Skip past any whitespace
                    while start < len(text) and text[start] in ' \t\n':
                        start += 1
            
            if start >= len(text):
                break
            
            end = start + self.chunk_size
            
            # Don't break in the middle of the text
            if end < len(text):
                # Priority 1: Check if we're in a code block - never break these
                if self._is_in_code_block(text, end):
                    # Try to find the end of the code block
                    code_end = self._find_code_block_end(text, end)
                    
                    # If code block end is reasonable (within 1000 chars), use it
                    if code_end > end and code_end - start < self.chunk_size + 1000:
                        end = code_end
                    else:
                        # Code block is too large or spans beyond reasonable distance
                        # Try to find where the code block started
                        search_region = text[start:end]
                        
                        # Count backticks to find opening of current block
                        backtick_positions = []
                        pos = 0
                        while True:
                            idx = search_region.find('```', pos)
                            if idx == -1:
                                break
                            backtick_positions.append(start + idx)
                            pos = idx + 3
                        
                        # If we have backticks, find the one that opens current block
                        if backtick_positions:
                            # The last backtick in our region should be the opening
                            # (since we know we're inside a code block at 'end')
                            code_block_start = backtick_positions[-1]
                            
                            # If code block starts far enough in, break before it
                            if code_block_start - start > self.chunk_size // 2:
                                # Break just before the code block
                                end = code_block_start
                            else:
                                # Code block started early, extend to include it all
                                if code_end > end:
                                    end = code_end
                        else:
                            # No backticks found, extend to end of code block
                            if code_end > end:
                                end = code_end
                
                # Priority 2: Check if we're in a table
                elif self._is_in_table(text, end):
                    # Try to find the end of the table
                    remaining = text[end:end+1000]
                    # Look for double newline (end of table section)
                    table_end = remaining.find('\n\n')
                    if table_end > 0 and table_end < 500:
                        end = end + table_end + 2
                    else:
                        # Find a better boundary
                        end = self._find_section_boundary(text, start, end)
                else:
                    # Find a good boundary that respects markdown structure
                    end = self._find_section_boundary(text, start, end)
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self.overlap if end < len(text) else len(text)
        
        return chunks

