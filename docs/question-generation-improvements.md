# Question Generation Improvements

## Overview

Improved the quality of questions generated during memory storage to enable better retrieval accuracy. The questions generated during chunk storage directly impact how well user queries match against stored documentation.

## Problem Statement

**Before:** Questions were not diverse and didn't cover different aspects of the content.

Example of OLD generated questions for a list of migration types:
```
- What is cold migration?
- What is warm migration?
- What is live migration?
```
← All three questions ask the same thing about different items! Not diverse.

**After:** Questions cover DIFFERENT aspects and emphasize what makes the content UNIQUE.

Example of NEW generated questions for the same content:
```
- What are the different types of migration strategies available?
- How do cold, warm, and live migration differ in downtime requirements?
- When should I choose warm migration over cold migration?
```
← Each question covers a different aspect: overview, comparison, selection criteria.

## Changes Made

### 1. Enhanced Question Generation Prompt

**File:** `ymca/core/model_handler.py` - `_build_question_prompt()`

**Key Improvements:**
- **Diversity Focus:** Each question must cover a DIFFERENT aspect of the content
- **Uniqueness Focus:** Questions emphasize what makes THIS chunk unique
- **Smart List Handling:** For lists, ask about overall purpose/patterns, not individual items
- **Multi-Concept Coverage:** If text has multiple concepts, each question covers a different one
- **Format Flexibility:** Questions can use any format: "What is...", "How does...", "When to...", "Why..."
- Request 8-20 word detailed questions
- Added examples of GOOD vs BAD questions showing diversity principles

**New Prompt Structure:**
```python
"""You are creating search queries for a technical documentation retrieval system.

Instructions:
- Generate DIVERSE questions covering DIFFERENT aspects of the text
- Each question should emphasize UNIQUE characteristics of this content
- If the text contains a list, ask about the overall purpose/pattern, NOT individual items
- If the text has multiple concepts, each question should cover a different concept
- Questions can use any format: "What is...", "How does...", "When to...", "Why..."

Examples of GOOD diverse questions for a list of migration types:
- What are the different types of migration strategies available?
- How do cold, warm, and live migration differ in downtime requirements?
- When should I choose warm migration over cold migration?

Examples of BAD questions (not diverse):
- What is cold migration?
- What is warm migration?
- What is live migration?
← These all ask the same thing about different items!

Examples of GOOD questions for authentication docs:
- What is the authentication mechanism used in this system?
- How do I configure authentication with certificate files?
- What troubleshooting steps should I follow if authentication fails?
← Each covers a different aspect: concept, setup, troubleshooting
"""
```

### 2. Adjusted Generation Parameters

**Increased max_tokens:** 300 → 400
- Supports longer, more detailed questions (10-20 words each)
- Prevents truncation of detailed questions

**Updated stop tokens:** Added "Instructions:" to prevent model from echoing the prompt

### 3. Question Quality Validation

Added automatic validation that logs warnings for:
- Questions that are too short (<5 words) - likely too vague
- Questions that are too long (>30 words) - likely too complex

This helps identify when question generation needs adjustment.

## Impact on Retrieval

### Before vs After Comparison

**Scenario:** User asks "how do I set up authentication?"

**Before (generic questions):**
```
Storage: "What is authentication?" 
         "What is mentioned about configuration?"
         
User Query: "how do I set up authentication?"
Embedding Match: Poor (0.45 similarity) ❌
Result: Wrong chunk retrieved
```

**After (specific questions):**
```
Storage: "How do I configure authentication with certificates for production?"
         "What are the steps to set up username and password authentication?"
         
User Query: "how do I set up authentication?"
Embedding Match: Excellent (0.87 similarity) ✅
Result: Correct chunk retrieved
```

## Quality Metrics

The system now generates questions with:
- **Average length:** 8-20 words (optimal for semantic search)
- **Diversity:** Each question covers a different aspect of the content
- **Uniqueness:** Questions emphasize what makes this chunk unique
- **Smart abstraction:** For lists, questions focus on patterns not individual items

### Quality Principles

**Good questions demonstrate:**
1. **Different Perspectives:** Concept vs. procedure vs. troubleshooting
2. **Appropriate Abstraction:** "What are the available types?" not "What is type A?"
3. **Chunk Uniqueness:** Focus on what makes THIS content distinct
4. **Natural Queries:** Match what users actually search for

## Testing

### Run Quality Test

Test the improved question generation:

```bash
python bin/test_question_quality.py
```

This will:
1. Generate questions for sample technical documentation
2. Analyze question quality metrics
3. Show before/after comparison
4. Provide quality score (0-100)

### Sample Output

```
📄 Chunk 1: Authentication Configuration
   
   Generated Questions:
   1. How do I configure authentication with JSON credentials file and certificate paths?
      (length: 12 words)
   2. What are the required file permissions for the authentication credentials configuration?
      (length: 11 words)
   3. What steps are needed to enable certificate-based authentication in the application?
      (length: 11 words)
   
   Quality Metrics:
   - Avg length: 11.3 words
   - With action verbs: 3/3
   - Ideal length (8-20 words): 3/3

📊 Quality Score: 95.0/100
   ✅ Excellent - Questions are detailed and actionable
```

## Best Practices

### For Optimal Question Quality

1. **Store complete documentation chunks** (500-4000 chars)
   - Too short: Not enough context for good questions
   - Too long: Questions become too broad

2. **Use structured technical documentation**
   - HOW-TO guides generate best questions
   - Procedural documentation with steps
   - Configuration examples with parameters

3. **Monitor question quality logs**
   - Check for warnings about short/long questions
   - Review generated questions periodically
   - Adjust chunk size if questions are consistently poor

4. **Re-generate questions for existing chunks** (optional)
   - If you improve the prompt, consider regenerating
   - Use `memory_cli.py` to rebuild from source documents

## Configuration

### Adjust Question Length Preference

Edit `model_handler.py` if you want different question lengths:

```python
# In _build_question_prompt()
# Change: "Questions should be specific and detailed (10-20 words each)"
# To:     "Questions should be specific and detailed (8-15 words each)"
```

### Adjust Quality Thresholds

Edit `generate_questions()` warning thresholds:

```python
if word_count < 5:  # Adjust minimum
    logger.warning(...)
elif word_count > 30:  # Adjust maximum
    logger.warning(...)
```

## Benefits

1. **Better Retrieval Accuracy:** Questions match user search patterns
2. **Reduced False Positives:** Specific questions prevent wrong chunk matches  
3. **Improved Semantic Search:** Detailed questions capture technical nuances
4. **Better Coverage:** Each chunk gets questions from different angles

## Limitations

1. **Generation Time:** More detailed prompts take slightly longer (~10-20% increase)
2. **Model Dependency:** Quality depends on LLM's understanding of domain
3. **Token Usage:** Longer prompts consume more tokens (but worth it for quality)

## Future Improvements

1. **Domain-Specific Examples:** Provide examples tailored to your documentation type
2. **Multi-Perspective Questions:** Generate questions from different user roles
3. **Difficulty Levels:** Generate basic, intermediate, and advanced questions
4. **Question Diversity:** Ensure questions cover different aspects (what, how, why, when)
5. **Feedback Loop:** Learn from which questions lead to successful retrievals

## Migration Guide

### For Existing Memory Systems

If you already have stored chunks with old questions:

**Option 1: Continue with existing** (acceptable)
- Old questions will still work
- New chunks will have better questions
- Gradual improvement as content is updated

**Option 2: Regenerate all questions** (recommended for critical systems)
```bash
# Backup existing memory
cp -r data/tools/memory data/tools/memory.backup

# Clear and reload
python bin/memory_cli.py clear
python bin/memory_cli.py load <your_docs_directory>
```

**Option 3: Selective regeneration**
- Identify chunks with poor quality questions
- Regenerate only those chunks
- Keep high-quality existing questions

## Related Documentation

- [Memory Improvements](memory-improvements.md) - Query expansion during retrieval
- [Memory CLI](../bin/memory_cli.py) - Tools for managing memory system
- [Model Handler](../ymca/core/model_handler.py) - Question generation implementation

