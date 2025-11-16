# Small Models That Use Tools: like a crow using a stick

We've long been fascinated by animals that use tools—a crow bending a wire to hook a treat, a sea otter using a rock to crack open a shell. We see in them a spark of ingenuity, an ability to transcend their physical limitations by leveraging an external object.

<figure>
<img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Stone_tool_use_by_a_capuchin_monkey.jpg" width="300" alt="Capuchin monkey using a stone tool to crack open nuts">
<figcaption><small>Capuchin monkey using stone tool. <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>, via <a href="https://commons.wikimedia.org/wiki/File:Stone_tool_use_by_a_capuchin_monkey.jpg">Wikimedia Commons</a></small></figcaption>
</figure>

In the world of artificial intelligence, we are witnessing a similar development. For years, the AI arms race was defined by a simple, brute-force metric: bigger is better. Models with hundreds of billions of parameters dominated the headlines. Running in parallel, a quieter evolution has been unfolding, one that may prove far more practical: the rise of the [Small Language Model (SLM)](https://en.wikipedia.org/wiki/Small_language_model).

Like that clever crow, these SLMs are demonstrating that you don't need massive scale to achieve useable results, you just need to know which tools to use and when to use them.

The question is no longer whether a massive, cloud-based AI can write a poem. The new, interesting question is whether a small, efficient model running entirely on your laptop can intelligently and reliably act on your behalf, reading logs, querying databases, and controlling applications.

## The Case for Local AI: Privacy, Speed, and Sustainability

But why run an AI locally in the first place, when large model can do the job much better? The most critical reason is privacy and data security. For industries like healthcare, finance, or law, sending sensitive patient or client data to a third-party cloud is often a non-starter. When the model runs on your own device, your data never leaves your machine, ensuring compliance and confidentiality. This local-first approach also eliminates cloud latency, providing instant responses for real-time applications, and works entirely offline, independent of internet connectivity.

Beyond individual use cases, there is a significant environmental benefit. The massive, cloud-based models that dominate the headlines are resource-intensive, consuming staggering amounts of electricity for training and deployment, as well as vast quantities of water for cooling. This has created a growing energy footprint. Small Language Models (SLMs) running on efficient local hardware are a direct solution.

The benefits are clear, but the challenge remains: how do you build a practical, local AI agent that can truly act on your behalf over time? A small model has inherent limitations, a narrow context window and a limited "world knowledge" baked into its parameters. To compensate, we must augment the small model with two critical external capabilities: a persistent, searchable memory, and a dynamic, scalable toolkit. Let's examine how each of these works.

## An Agent's Memory: The Digital Sleep Cycle

An agent's first limitation is memory, and the solution is deeply analogous to human memory consolidation, the process that occurs during sleep. While we're awake, our brain rapidly stores new experiences. But it's during sleep, in "offline" processing, that these temporary memories are replayed, filtered, and integrated into our long-term storage. AI systems face a similar challenge: a model's finite "context window" is like short-term memory; it can't remember your first conversation from last week. To solve this, new architectures are being built to mimic this human "sleep" cycle.

A foundational strategy is [Retrieval-Augmented Generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation). In this model, an agent retrieves chunks of information from a static, external "textbook"—a vector database of your documents. This is a massive leap, but it is fundamentally a read-only system. It's effective for looking up facts but fails to capture the episodic flow of a long-term relationship.

A more sofisticated aprouch, seen in architectures like Mem0, focuses on creating a dynamic "diary". This system introduces a crucial read-write capability, specifically an "Update Phase" that mimics the brain's offline consolidation. As a conversation unfolds, the agent is given function-calling tools like ADD, UPDATE, and DELETE to actively curate its own memory. When you provide new information, the model "sleeps" on it, deciding if it's a new fact (ADD), a correction to an old one (UPDATE), or a contradiction that must be removed (DELETE). This process integrates isolated past experiences into a coherent and non-redundant memory.

The most advanced form of this, Mem0g, takes it a step further by building a graph-based memory. Instead of just storing "text," it stores relationships—nodes (like "Project_Alpha," "Alice") and edges (like "is_assigned_to," "is_due_on"). This allows for complex, multi-hop reasoning, letting the agent answer not just "What did we say about Alice?" but "Who works with Alice on projects due next month?".

## An Agent's Toolkit: The Mental Workspace

An agent's second limitation is action, and this challenge mirrors the "matching problem" in human cognition. Humans possess a vast suite of "cognitive tools" and mental models, but we don't load all of them into our conscious mind at once. Instead, we use analogical reasoning to assess a new problem and recall only the relevant past experiences or tools to solve it; irrelevant ones are distracting. If you're cooking, you mentally pull up your "kitchen toolset," not your "car repair toolset."

AI agents face a digital version of this. Naive tool-use frameworks tried to solve this by stuffing every available tool definition, every API, every function, into the model's system prompt. This "prompt bloat" is the digital equivalent of trying to work in a room cluttered with every tool you own. Just as a human engineer would be confused by a bloated tool set, an AI agent also loses focus. The solution, therefore, is to teach the AI to do what we do instinctively: create a clean, task-specific "mental workspace" containing only the tools it needs for the job at hand.

The strategies for managing this mirror the approaches to memory. One effective strategy is Retrieval-Augmented Tool Selection (RAG-TS). Just like RAG for documents, this approach treats the tool library itself as a searchable database. When a task arrives, the agent first performs a semantic search to find the top-k relevant tools, then loads only those into its context for execution.

A more advanced strategy, Dynamic Context Tuning (DCT), is designed for evolving environments, like a smart home where new devices are added. Instead of just retrieving text, DCT can dynamically load tiny, specialized updates that "tune" the model on the fly, instantly teaching it how to use a previously unseen tool without any retraining.

## YMCA: A Practical Implementation

This brings us to [YMCA](https://github.com/yaacov/ymca) (Yaacov's MCP Agent), a concrete implementation of these principles. YMCA runs entirely on your laptop, demonstrating that a powerful AI assistant can be fast, private, and resource-efficient.

At its core, YMCA uses IBM's [Granite Tiny](https://www.ibm.com/granite), a small language model specifically trained for tool use and reasoning. Granite Tiny is not designed to memorize vast amounts of factual knowledge. Instead, it excels at understanding instructions, decomposing complex requests into steps, and deciding which tools to invoke. Its primary skill is orchestration: knowing when to think, when to remember, and when to act.

For memory, YMCA implements a simplified approach inspired by Mem0. Rather than the full read-write memory system with ADD, UPDATE, and DELETE operations, YMCA focuses on the read path for simplicity and efficiency. The agent has access to a memory tool that allows it to retrieve relevant information from past interactions. When you ask YMCA about a previous conversation or project detail, it calls `retrieve_from_memory(query)`, searching a vector database of your interaction history. This read-only "RAG-as-a-Tool" approach provides persistent memory without the computational overhead of active memory curation, keeping the local footprint minimal.

For its toolkit, YMCA leverages the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). MCP is an open standard that acts as a universal interface for AI tools, much like USB-C standardized physical connections. At startup, YMCA connects to configured MCP servers and discovers all available tools. For each query, YMCA uses semantic search to select only the most relevant tools and includes just those in the system prompt. This keeps Granite Tiny focused, seeing only what it needs rather than being overwhelmed by dozens of irrelevant tool definitions. The model then decides which tools to call, executes them via MCP, and synthesizes the answer.

The result is a local agent that stays small and fast. Its "brain" (Granite Tiny) is a compact orchestrator. Its memory is a tool call away. And its capabilities are virtually limitless, dynamically discovered through the MCP ecosystem. YMCA demonstrates that the small model, can be the efficient conductor of a powerful, privacy preserving agent system.