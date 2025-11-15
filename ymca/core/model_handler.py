"""
Model Handler - Manages LLM for question generation and other tasks.
"""

from pathlib import Path
from typing import List, Optional
import logging
import gc
import time

from llama_cpp import Llama
from .question_parser import parse_questions_from_text

logger = logging.getLogger(__name__)


class ModelHandler:
    """Handle LLM operations for memory tool."""
    
    def __init__(self, model_path: str, n_ctx: int = 32768, n_threads: int = 4, n_gpu_layers: int = -1):
        """
        Initialize the model handler.
        
        Args:
            model_path: Path to GGUF model (required)
            n_ctx: Context size (default: 32768, Granite 4.0 supports up to 128K but GGUF may have limits)
            n_threads: Number of threads (default: 4)
            n_gpu_layers: Number of layers to offload to GPU (default: -1 = all layers, 0 = CPU only)
                         Falls back to CPU if GPU support is not available
        """
        if not model_path:
            raise ValueError("model_path is required")
        
        if not Path(model_path).exists():
            raise ValueError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model: {Path(model_path).name}...")
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,  # -1 = all layers to GPU, 0 = CPU only
            verbose=False
        )
        
        # Log acceleration info
        gpu_info = "GPU acceleration enabled" if n_gpu_layers != 0 else "CPU only"
        if n_gpu_layers != 0:
            gpu_info += " (will fall back to CPU if GPU unavailable)"
        logger.info(f"✓ Model loaded (context: {n_ctx} tokens, {gpu_info})")
    
    def reset_state(self, deep_clean: bool = False):
        """
        Reset the model's KV cache to clear any accumulated state.
        
        Args:
            deep_clean: If True, also run garbage collection and brief pause
        """
        try:
            # Reset llama.cpp state
            self.llm.reset()
            logger.debug("Model state reset")
        except AttributeError:
            # Fallback for older llama-cpp-python versions
            logger.debug("Model reset not available")
            pass
        
        # Deep clean if requested
        if deep_clean:
            # Force garbage collection to free memory
            gc.collect()
            # Brief pause to let system settle
            time.sleep(0.1)
            logger.debug("Deep clean completed")
    
    def generate_questions(self, chunk: str, num_questions: int = 3) -> List[str]:
        """
        Generate questions that can be answered by the chunk.
        
        Uses the LLM to generate relevant questions about the text content.
        Fails explicitly if the LLM cannot generate the requested number of questions.
        
        Args:
            chunk: Text chunk to generate questions about
            num_questions: Number of questions to generate (default: 3)
            
        Returns:
            List of generated questions
            
        Raises:
            ValueError: If parsing fails or not enough questions generated
            Exception: If LLM call fails
        """
        prompt = self._build_question_prompt(chunk, num_questions)
        
        # Generate questions via LLM
        # Note: max_tokens needs to be high enough for detailed questions
        # Detailed questions (10-20 words) ~= 15-30 tokens each
        # For 2-3 questions: 40-90 tokens + overhead = ~150-200 tokens needed
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,  # Increased to support detailed questions (10-20 words each)
            temperature=0.7,  # Balanced between creativity and consistency
            stop=["\n\n", "Text:", "---", "Instructions:"],  # Stop tokens to prevent rambling
            repeat_penalty=1.1  # Reduce repetition
        )
        
        # Extract and parse generated text (will raise ValueError if insufficient)
        generated_text = response['choices'][0]['message']['content'].strip()
        questions = parse_questions_from_text(generated_text, num_questions)
        
        # Validate and log question quality
        for i, q in enumerate(questions, 1):
            word_count = len(q.split())
            if word_count < 5:
                logger.warning(f"Question {i} is very short ({word_count} words): {q}")
            elif word_count > 30:
                logger.warning(f"Question {i} is very long ({word_count} words): {q}")
        
        return questions
    
    def _build_question_prompt(self, chunk: str, num_questions: int) -> str:
        """
        Build prompt for question generation optimized for retrieval.
        
        Creates questions that users would naturally ask when looking for this information.
        Focuses on HOW-TO, configuration, troubleshooting, and procedural aspects.
        
        Args:
            chunk: Text chunk
            num_questions: Number of questions to generate
            
        Returns:
            Formatted prompt string
        """
        return f"""Given the following technical documentation, generate exactly {num_questions} questions about this content.

Text:
{chunk}

Instructions:
- Generate exactly {num_questions} DIVERSE questions covering DIFFERENT aspects of the text
- Each question should emphasize UNIQUE characteristics of this content
- If the text contains a list, ask about the overall purpose/pattern, NOT individual items
- If the text has multiple concepts, each question should cover a different concept
- Questions should be specific to what makes THIS content unique
- Questions should be specific and detailed (8-20 words each)
- Each question must be on a separate line
- Do not number the questions
- Each question must end with a question mark

Questions:"""
    

    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False
        )
        
        return response['choices'][0]['text'].strip()
    
    def chat(
        self, 
        messages: List[dict], 
        max_tokens: int = 200, 
        temperature: float = 0.7,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None
    ) -> dict:
        """
        Chat completion with optional tool calling support.
        
        Args:
            messages: List of message dictionaries with role and content
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tools: Optional list of tool definitions for function calling
            tool_choice: Tool selection mode ("auto", "none", or specific tool name)
            
        Returns:
            Full response dictionary from llama_cpp
        """
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add tools if provided
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        
        return self.llm.create_chat_completion(**kwargs)
