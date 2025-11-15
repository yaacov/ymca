# Hallucination Prevention in RAG Systems

## The Problem

Even with perfect memory retrieval, LLMs can hallucinate by:
1. **Inventing questions** - Creating different questions than the user asked
2. **Adding fake features** - Listing capabilities not in the retrieved documentation
3. **Wrong format** - Using Q&A format when not requested
4. **Ignoring context** - Not using the tool results that were retrieved

## Example of Hallucination

**User asks:** "Can you help me create a migration plan?"

**System retrieves:** ✅ Correct documentation about `kubectl mtv create plan`

**LLM hallucinates:**
```
Question: Can you provide a brief summary of the key features...
Answer: kubectl-mtv includes:
  1. AI-Assisted Migration ← MADE UP!
  2. Machine learning algorithms ← MADE UP!
  3. ...
```

❌ **Problems:**
- Invented a different question
- Ignored the actual user question
- Added features not in the documentation
- Used Q&A format unprompted

## The Solution: Strict System Prompt

**File:** `ymca/chat/api.py`

### Base System Message (Lines 51-67)

```python
self.base_system_message = system_message or (
    "You are a technical assistant. Answer the USER'S EXACT QUESTION using ONLY information from tool results.\n\n"
    "CRITICAL RULES:\n"
    "1. Answer ONLY the user's actual question - never invent your own question\n"
    "2. Use ONLY facts explicitly stated in tool results - no assumptions or inventions\n"
    "3. If tool results don't contain the answer, say 'I don't have that information'\n"
    "4. NEVER add features, capabilities, or details not in the tool results\n"
    "5. Keep answers SHORT: 1-3 sentences, then examples if helpful\n\n"
    "TOOL QUERY BEST PRACTICES:\n"
    "When calling retrieve_memory or similar search tools, use detailed, specific queries:\n"
    "✓ GOOD: 'how to create authentication configuration step by step'\n"
    "✓ GOOD: 'troubleshooting database connection errors'\n"
    "✗ BAD: 'authentication' (too vague)\n"
    "✗ BAD: 'configuration' (too general)\n"
    "Include action words (how to, configure, troubleshoot, create, setup) and specific technical terms.\n\n"
    "FORMAT: Just answer directly. Don't create fake Q&A format. Don't add 'Question:' or 'Answer:' labels."
)
```

### Enhanced Instructions with Tools (Lines 220-235)

```python
CRITICAL INSTRUCTIONS:

**Tool Calling:**
- Output ONLY the <tool_call> block with properly formatted JSON
- Do NOT add any text before or after the tool call
- Wait for the result before responding

**Answering Questions:**
- Answer the USER'S ACTUAL QUESTION - do not invent different questions
- Use ONLY information explicitly stated in tool results - no additions or assumptions
- If information isn't in the tool results, say 'I don't have that information in the documentation'
- Answer in 1-3 SHORT sentences, then add examples if helpful
- DO NOT format as 'Question: ... Answer: ...' - just answer directly
- DO NOT list features or capabilities not mentioned in the tool results

**Remember:** Answer the actual question. Tool results only. No hallucinations. Be brief.
```

## Key Prevention Strategies

### 1. Explicit "Actual Question" Reminder

**Before:**
```
"Base your answers on information from tool results"
```

**After:**
```
"Answer the USER'S EXACT QUESTION using ONLY information from tool results"
"Answer ONLY the user's actual question - never invent your own question"
```

### 2. Negative Examples

Tell the LLM what NOT to do:

```
"NEVER add features, capabilities, or details not in the tool results"
"DO NOT format as 'Question: ... Answer: ...' - just answer directly"
```

### 3. Fallback Response

Provide a safe default when information is missing:

```
"If information isn't in the tool results, say 'I don't have that information in the documentation'"
```

### 4. Format Enforcement

Explicitly prevent unwanted formats:

```
"FORMAT: Just answer directly. Don't create fake Q&A format. Don't add 'Question:' or 'Answer:' labels."
```

### 5. Repetition for Emphasis

Repeat the core constraint multiple times:
- In base message: "using ONLY information from tool results"
- In critical rules: "Use ONLY facts explicitly stated"
- In remember section: "Tool results only. No hallucinations."

## Expected Behavior

### ✅ Correct Response

**User:** "Can you help me create a migration plan?"

**Tool retrieves:** Documentation about `kubectl mtv create plan`

**LLM responds:**
```
To create a migration plan, use `kubectl mtv create plan <name>` with these key flags:
- `--source`: Source provider name
- `--vms`: VM names or query
- `--network-mapping` and `--storage-mapping`: Resource mappings

Example:
kubectl mtv create plan my-plan --source vsphere-source --vms "web-01,db-01"
```

✅ Answers the actual question  
✅ Uses only information from documentation  
✅ No invented features  
✅ Direct format, no fake Q&A  

### ❌ Hallucinated Response (Prevented)

```
Question: Can you provide a brief summary...  ← INVENTED QUESTION
Answer: kubectl-mtv includes:
  1. AI-Assisted Migration ← FAKE FEATURE
  2. Machine learning algorithms ← FAKE FEATURE
```

## Testing Hallucination Prevention

### Test Cases

1. **Question Substitution Test**
   - User asks: "How do I X?"
   - Verify LLM doesn't create: "Question: What are the features of Y?"

2. **Feature Invention Test**
   - Retrieve documentation with 5 features
   - Verify LLM doesn't list 10 features

3. **Format Adherence Test**
   - User asks direct question
   - Verify LLM doesn't add "Question:" and "Answer:" labels

4. **Missing Information Test**
   - User asks about feature not in docs
   - Verify LLM says "I don't have that information" instead of inventing

### Manual Testing

```bash
# Start chat app with memory
python bin/chat_app.py --memory

# Test queries that commonly trigger hallucinations:
1. "What features does X have?"
2. "Tell me about Y capabilities"
3. "How do I configure Z?"
4. "What's the difference between A and B?"

# Verify:
- Answers use ONLY retrieved information
- No invented questions
- No made-up features
- Direct answer format
```

## Common Hallucination Patterns

### Pattern 1: Question Invention

**Trigger:** Open-ended user question  
**Symptom:** LLM creates "Question: ... Answer: ..." format  
**Prevention:** "DO NOT format as 'Question: ... Answer: ...'"

### Pattern 2: Feature Expansion

**Trigger:** Listing capabilities  
**Symptom:** Lists more features than in documentation  
**Prevention:** "NEVER add features, capabilities, or details not in the tool results"

### Pattern 3: Assumption Filling

**Trigger:** Incomplete information in retrieval  
**Symptom:** LLM "fills in" missing details with assumptions  
**Prevention:** "Use ONLY facts explicitly stated in tool results"

### Pattern 4: Context Ignoring

**Trigger:** Strong prior knowledge about topic  
**Symptom:** Answers from training data instead of tool results  
**Prevention:** "Answer using ONLY information from tool results"

## Monitoring for Hallucinations

### Red Flags

Watch for these patterns in responses:

1. **Different question:** Response starts with "Question:" when user didn't format as Q&A
2. **Unsourced claims:** Statements not traceable to tool results
3. **Excessive detail:** More specific than the retrieved documentation
4. **Lists with round numbers:** "Here are 5/10/15 features..." when docs have 3
5. **Hedging language:** "typically", "usually", "often" when docs state facts

### Logging

Enable debug logging to see tool results:

```python
logging.basicConfig(level=logging.DEBUG)
```

Check that LLM responses align with tool output logged as:
```
DEBUG - Tool result (14852 chars): <actual documentation>
```

## Model-Specific Considerations

### Small Models (e.g., Granite 4B)

- More prone to hallucination due to limited reasoning
- Need **stronger, more explicit** instructions
- Benefit from **repetition** of key constraints
- May require **shorter context** to stay focused

### Recommendations for Small Models

1. **Shorter system prompts:** Break into critical points
2. **Explicit negatives:** Tell what NOT to do
3. **Safe fallbacks:** Provide default responses
4. **Format enforcement:** Specify exact output format
5. **Context limitation:** Keep tool results concise

## Integration with Memory System

This hallucination prevention completes the full RAG pipeline:

```
User Question
     ↓
[LAYER 1: Query Formulation]
  → System prompt guides LLM to create good queries
  → "use detailed, specific queries"
     ↓
[LAYER 2: Query Expansion]
  → Auto-expand vague queries
  → "authentication" → "how to configure authentication"
     ↓
[LAYER 3: Retrieval]
  → Search with good questions
  → Match against well-generated questions in storage
     ↓
[LAYER 4: Response Generation] ← THIS DOCUMENT
  → Strict instructions prevent hallucination
  → "Use ONLY information from tool results"
     ↓
Accurate Answer
```

## Related Documentation

- [Memory System Complete Improvements](memory-system-complete-improvements.md) - Full RAG pipeline
- [Question Generation Improvements](question-generation-improvements.md) - Storage side
- [Memory Improvements](memory-improvements.md) - Retrieval side
- [Question Generation Principles](question-generation-principles.md) - Quality guidelines

## Summary

**Hallucination prevention requires:**
1. ✅ Explicit instructions to answer the ACTUAL question
2. ✅ Repeated emphasis on using ONLY tool results
3. ✅ Negative examples (what NOT to do)
4. ✅ Safe fallback responses
5. ✅ Format enforcement
6. ✅ Multiple reminders throughout prompt

**The key insight:** Small models need **very explicit, repetitive guidance** to avoid hallucinations, especially when they have strong prior knowledge that conflicts with retrieved information.

