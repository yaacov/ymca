# Question Generation Principles

## Core Philosophy

Generate **diverse questions** that emphasize the **unique characteristics** of each chunk.

## Key Principles

### 1. Diversity Over Repetition

**Bad (Repetitive):**
```
Chunk: "There are three migration types: cold, warm, and live..."

Questions:
- What is cold migration?
- What is warm migration?  
- What is live migration?
```
❌ All three questions use the same pattern for different items!

**Good (Diverse):**
```
Chunk: "There are three migration types: cold, warm, and live..."

Questions:
- What are the different types of migration strategies available?
- How do cold, warm, and live migration differ in downtime requirements?
- When should I choose warm migration over cold migration?
```
✅ Each question covers a different aspect: overview, comparison, selection criteria

### 2. Emphasize Uniqueness

Questions should focus on what makes **THIS** chunk unique, not generic facts.

**Bad (Generic):**
```
Chunk: "To configure authentication, create a JSON credentials file with 
'user' and 'password' fields. Place in config directory with chmod 400 permissions..."

Questions:
- What is authentication?
- What is a JSON file?
- What are file permissions?
```
❌ These are general questions, not specific to this content!

**Good (Unique):**
```
Chunk: "To configure authentication, create a JSON credentials file with 
'user' and 'password' fields. Place in config directory with chmod 400 permissions..."

Questions:
- What is the format of the authentication credentials JSON file?
- Where should the authentication configuration file be placed and what permissions are needed?
- What fields are required in the authentication credentials file?
```
✅ These questions are specific to the unique details in this chunk!

### 3. Smart List Handling

For lists, ask about the **overall purpose/pattern**, not individual items.

**Bad (Item-by-Item):**
```
Chunk: Lists 5 troubleshooting steps for connection errors

Questions:
- What is step 1 for troubleshooting?
- What is step 2 for troubleshooting?
- What is step 3 for troubleshooting?
```
❌ Fragmenting a cohesive list into separate pieces!

**Good (Holistic):**
```
Chunk: Lists 5 troubleshooting steps for connection errors

Questions:
- What are the troubleshooting steps for connection timeout errors?
- How should I diagnose and resolve connection failures?
- What is the recommended troubleshooting workflow for network connectivity issues?
```
✅ Questions treat the list as a cohesive procedure!

### 4. Multi-Concept Coverage

If a chunk has multiple concepts, each question should cover a **different** concept.

**Bad (Single Concept):**
```
Chunk: Covers authentication setup, configuration files, and troubleshooting

Questions:
- How do I set up authentication?
- What are the authentication setup steps?
- How to configure authentication system?
```
❌ All three focus only on setup!

**Good (Multi-Concept):**
```
Chunk: Covers authentication setup, configuration files, and troubleshooting

Questions:
- How do I set up authentication with credentials files?
- What is the structure and location of authentication configuration files?
- What troubleshooting steps should I follow if authentication fails?
```
✅ Setup, configuration, troubleshooting - all covered!

### 5. Natural Query Formats

Use any question format that users naturally search for.

**All Valid:**
- "What is..." - Conceptual understanding
- "How does..." - Mechanism/process
- "How do I..." - Procedure/action
- "When to..." - Decision/timing
- "Why..." - Reasoning/rationale
- "Which..." - Selection/comparison
- "Where..." - Location/placement

**Example:**
```
- What is the authentication mechanism used in this system?
- How does certificate-based authentication work?
- How do I configure authentication credentials?
- When should I use token authentication vs certificates?
- Why is file permission 400 required for credentials?
- Which authentication method is recommended for production?
- Where should I place the authentication configuration file?
```

All of these are good questions if they match the content!

## Quality Checklist

For each generated question set, verify:

- [ ] **Diversity:** Do questions cover different aspects?
- [ ] **Uniqueness:** Do questions focus on this chunk's unique content?
- [ ] **Abstraction:** For lists, do questions treat them holistically?
- [ ] **Coverage:** Do questions cover all major concepts in the chunk?
- [ ] **Naturalness:** Would users actually search using these questions?

## Examples by Content Type

### Configuration Documentation

```
Chunk: Step-by-step database configuration guide

Good Questions:
- What are the configuration steps for database setup?
- What connection parameters are required in the database configuration?
- How do I validate that the database configuration is correct?
```

### Comparison/List Documentation

```
Chunk: Compares storage options: local, NFS, S3

Good Questions:
- What are the available storage backend options?
- How do local, NFS, and S3 storage backends differ in performance?
- When should I choose S3 storage over NFS?
```

### Troubleshooting Documentation

```
Chunk: Debugging connection timeout errors

Good Questions:
- What are the common causes of connection timeout errors?
- How do I diagnose and resolve connection timeout issues?
- What diagnostic commands should I run for connection troubleshooting?
```

### API/Reference Documentation

```
Chunk: CreatePlan API parameters and options

Good Questions:
- What parameters are required for the CreatePlan API?
- What optional configuration options does CreatePlan support?
- How do I specify VM selection criteria in CreatePlan?
```

## Anti-Patterns to Avoid

### ❌ Sequential Listing
```
- What is the first step?
- What is the second step?
- What is the third step?
```

### ❌ Definition Overload
```
- What is X?
- What is Y?
- What is Z?
```

### ❌ Same Question Different Nouns
```
- How do I configure cold migration?
- How do I configure warm migration?
- How do I configure live migration?
```

### ❌ Too Generic
```
- What does this describe?
- What is mentioned here?
- What information is provided?
```

## Implementation

These principles are implemented in:
- **File:** `ymca/core/model_handler.py`
- **Method:** `_build_question_prompt()`
- **Testing:** `bin/test_question_quality.py`

## Measuring Quality

**Diversity Score:** Number of different question types (what, how, when, why, which)
- Target: 3+ different types for a set of 3 questions

**Uniqueness:** Questions reference specific technical details from the chunk
- Manual review: Do questions mention specific parameters, steps, or details?

**Abstraction:** For list content, questions don't itemize
- Check: No "What is item 1?" type questions

## Benefits

1. **Better Retrieval:** Questions match user search patterns across different perspectives
2. **Reduced Duplication:** Each question adds unique value
3. **Comprehensive Coverage:** All aspects of content are searchable
4. **Natural Language:** Users search the way questions are phrased

## See Also

- [Question Generation Improvements](question-generation-improvements.md) - Implementation details
- [Complete Memory System Improvements](memory-system-complete-improvements.md) - Full context

