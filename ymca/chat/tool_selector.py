"""Tool selector using semantic search to find relevant tools."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Set, TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .tool import Tool

logger = logging.getLogger(__name__)


class ToolSelector:
    """
    Selects most relevant tools for a query using semantic similarity.
    
    Uses semantic summaries (declarative statements) to represent tool capabilities,
    providing better matching against user queries than question-based approaches.
    This helps keep the system prompt short by only including tools that are
    likely to be useful for the current query.
    """
    
    def __init__(self, embedder=None, model_handler=None, cache_dir: Optional[str] = None, num_queries: int = 5):
        """
        Initialize tool selector.
        
        Args:
            embedder: Embedder for semantic search (required for tool selection).
            model_handler: Optional model handler for generating semantic summaries
            cache_dir: Directory to store tool index cache (default: data/tools/selector)
            num_queries: Number of semantic summaries to generate per tool (default: 5)
        """
        self.embedder = embedder
        self.model_handler = model_handler
        self.num_queries = num_queries
        self.tool_queries: Dict[str, List[str]] = {}  # Cache generated summaries (kept name for compatibility)
        self.tool_descriptions: Dict[str, str] = {}  # Store tool descriptions
        self.query_embeddings: Dict[str, List[tuple]] = {}  # Maps tool_name -> [(summary, embedding), ...]
        self.always_include: Set[str] = set()
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = "data/tools/selector"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing tool index if available
        self._load_existing_index()
        
    def set_always_include(self, tool_names: List[str]):
        """
        Set tools that should always be included.
        
        Args:
            tool_names: List of tool names to always include
        """
        self.always_include = set(tool_names)
    
    def _load_existing_index(self):
        """
        Load existing tool index from disk if available.
        
        This preserves previously generated semantic summaries and descriptions,
        avoiding regeneration on every startup.
        """
        index_file = self.cache_dir / "tool_index.json"
        if not index_file.exists():
            logger.debug(f"No existing tool index found at {index_file}")
            return
        
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            tools_data = index_data.get("tools", {})
            for tool_name, tool_data in tools_data.items():
                # Load queries and descriptions
                example_queries = tool_data.get("example_queries", [])
                description = tool_data.get("description", "")
                
                if example_queries:
                    self.tool_queries[tool_name] = example_queries
                    logger.debug(f"Loaded {len(example_queries)} queries for tool '{tool_name}'")
                
                if description:
                    self.tool_descriptions[tool_name] = description
            
            logger.info(f"Loaded existing tool index with {len(self.tool_queries)} tools from {index_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load existing tool index: {e}")
    
    def _generate_example_queries(self, tool_name: str, tool_description: str) -> List[str]:
        """
        Generate semantic summaries describing the tool's capabilities.
        
        Creates declarative statements that capture what the tool does, when to use it,
        and what capabilities it provides. These summaries are used for semantic matching
        against user queries to select relevant tools.
        
        Args:
            tool_name: Name of the tool
            tool_description: Description of what the tool does
            
        Returns:
            List of semantic summaries (configured by num_queries)
        """
        if not self.model_handler:
            return []
        
        # Check cache first
        if tool_name in self.tool_queries:
            return self.tool_queries[tool_name]
        
        try:
            # Reset KV cache to prevent contamination between tool query generations
            if hasattr(self.model_handler, 'reset_state'):
                self.model_handler.reset_state()
            
            prompt = (
                f"Tool: {tool_name}\n"
                f"Description: {tool_description}\n\n"
                f"Generate {self.num_queries} comprehensive semantic summaries that describe this tool's capabilities and use cases.\n\n"
                f"Instructions:\n"
                f"- Generate DECLARATIVE statements (not questions) that describe DIFFERENT aspects of the tool\n"
                f"- Each statement should be a complete, self-contained description\n"
                f"- Focus on WHAT the tool does, WHEN to use it, WHY it's useful\n"
                f"- Include specific capabilities, parameters, and use cases\n"
                f"- Each statement should capture unique information that would match user queries\n"
                f"- Statements should be rich and detailed (15-30 words each)\n"
                f"- Each statement must be on a separate line\n"
                f"- Do not number the statements\n"
                f"- Use present tense and be factual\n\n"
                f"Semantic summaries:"
            )
            
            # Calculate max_tokens based on number of summaries (roughly 50 tokens per summary)
            max_tokens = max(150, self.num_queries * 50)
            
            response = self.model_handler.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2,  # Low temperature for accurate tool descriptions
                stop=["\n\n", "Tool:", "---", "Instructions:"]
            )
            
            content = response['choices'][0]['message']['content'].strip()
            summaries = [s.strip() for s in content.split('\n') if s.strip()][:self.num_queries]
            
            # Cache the result
            self.tool_queries[tool_name] = summaries
            logger.debug(f"Generated {len(summaries)} semantic summaries for tool '{tool_name}'")
            return summaries
            
        except Exception as e:
            logger.debug(f"Failed to generate semantic summaries for tool '{tool_name}': {e}")
            return []
    
    def index_tools(self, tools: Dict[str, 'Tool']):
        """
        Index tools for semantic search using semantic summaries.
        
        Generates declarative statements describing each tool's capabilities
        and embeds them for matching against user queries. Each tool gets
        multiple semantic summaries covering different aspects and use cases.
        
        For tools already in the index, uses existing summaries instead of regenerating.
        
        Args:
            tools: Dictionary of tool name -> Tool object
        """
        if not self.embedder:
            logger.warning("No embedder provided, tool selection will not work")
            return
        
        new_tools = 0
        existing_tools = 0
        
        logger.info(f"Indexing {len(tools)} tools for semantic selection")
        
        for name, tool in tools.items():
            # Store tool description
            self.tool_descriptions[name] = tool.description
            
            # Check if we already have summaries for this tool
            if name in self.tool_queries and self.tool_queries[name]:
                semantic_summaries = self.tool_queries[name]
                existing_tools += 1
                logger.debug(f"Using existing {len(semantic_summaries)} summaries for tool '{name}'")
            else:
                # Generate new semantic summaries for this tool
                semantic_summaries = []
                if self.model_handler:
                    semantic_summaries = self._generate_example_queries(name, tool.description)
                    if semantic_summaries:
                        new_tools += 1
                        logger.debug(f"Generated {len(semantic_summaries)} new summaries for tool '{name}'")
            
            # Embed each semantic summary individually
            summary_embed_list = []
            for summary in semantic_summaries:
                try:
                    summary_embedding = self.embedder.embed_single(summary)
                    summary_embed_list.append((summary, summary_embedding))
                except Exception as e:
                    logger.warning(f"Failed to embed summary for tool '{name}': {e}")
            
            if summary_embed_list:
                self.query_embeddings[name] = summary_embed_list
        
        logger.info(f"Indexed {len(self.query_embeddings)} tools: {existing_tools} existing, {new_tools} new")
        
        # Save the index to a file for inspection
        self.save_index()
    
    def save_index(self):
        """
        Save tool index to human-readable JSON files.
        
        Saves only queries and metadata in human-readable format.
        Embeddings are kept in memory only (not persisted to disk).
        """
        try:
            # Get embedding dimension from first query embedding if available
            embedding_dim = 0
            if self.query_embeddings:
                first_tool_queries = next(iter(self.query_embeddings.values()))
                if first_tool_queries:
                    embedding_dim = len(first_tool_queries[0][1])
            
            # Build human-readable index with queries only (no embeddings)
            index_data = {
                "metadata": {
                    "num_tools": len(self.query_embeddings),
                    "embedding_dim": embedding_dim,
                    "note": "Embeddings are stored in memory only, not persisted to disk"
                },
                "tools": {}
            }
            
            for tool_name in sorted(self.query_embeddings.keys()):
                # Collect query info (text and norm only, no embedding vectors)
                query_info = []
                for query_text, query_embed in self.query_embeddings[tool_name]:
                    query_info.append({
                        "query": query_text,
                        "embedding_norm": float(np.linalg.norm(query_embed))
                    })
                
                tool_data = {
                    "description": self.tool_descriptions.get(tool_name, ""),
                    "example_queries": self.tool_queries.get(tool_name, []),
                    "query_stats": query_info
                }
                index_data["tools"][tool_name] = tool_data
            
            # Save human-readable JSON (queries only, no embedding vectors)
            index_file = self.cache_dir / "tool_index.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved tool index to {index_file} (queries only, no embeddings)")
            
            # Optionally save embeddings in binary format for persistence (if needed in future)
            # This is commented out since embeddings are regenerated on each startup
            # embeddings_file = self.cache_dir / "tool_embeddings.npy"
            # np.save(embeddings_file, embedding_data)
            
        except Exception as e:
            logger.warning(f"Failed to save tool index: {e}")
    
    def select_tools(
        self,
        query: str,
        tools: Dict[str, 'Tool'],
        max_tools: int = 3
    ) -> List[str]:
        """
        Select most relevant tools for a query.
        
        Args:
            query: User query
            tools: All available tools
            max_tools: Maximum number of tools to select
            
        Returns:
            List of selected tool names
        """
        # Always include specified tools
        selected = list(self.always_include)
        remaining_slots = max_tools - len(selected)
        
        if remaining_slots <= 0:
            return selected[:max_tools]
        
        # Get candidate tools (excluding always-included ones)
        candidates = {name: tool for name, tool in tools.items() if name not in self.always_include}
        
        if not candidates:
            return selected
        
        # Use semantic search if embedder and query embeddings available
        if self.embedder and self.query_embeddings:
            try:
                selected_candidates = self._semantic_select(query, candidates, remaining_slots)
                selected.extend(selected_candidates)
            except Exception as e:
                logger.warning(f"Semantic selection failed: {e}")
        else:
            logger.warning("No embedder or query embeddings available for tool selection")
        
        logger.debug(f"Selected {len(selected)} tools for query: {', '.join(selected)}")
        return selected
    
    def _semantic_select(self, query: str, tools: Dict[str, 'Tool'], max_tools: int) -> List[str]:
        """
        Select tools using semantic similarity to individual query embeddings.
        
        Algorithm:
        1. Compare user query against all individual query embeddings from all tools
        2. Sort all queries by similarity (best matches first)
        3. Pick first N unique tools from the sorted list
        
        This ensures diverse tool selection - even if one tool has multiple
        highly similar queries, we still get different tools in the results.
        """
        # Embed the user query
        query_embedding = self.embedder.embed_single(query)
        
        # Collect all (tool_name, query_text, similarity) tuples
        query_matches = []
        for tool_name in tools.keys():
            if tool_name in self.query_embeddings:
                for query_text, query_embed in self.query_embeddings[tool_name]:
                    # Cosine similarity
                    similarity = np.dot(query_embedding, query_embed) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(query_embed)
                    )
                    query_matches.append((tool_name, query_text, similarity))
        
        # Sort by similarity (best first)
        query_matches.sort(key=lambda x: x[2], reverse=True)
        
        # Pick first N unique tools from sorted list
        selected_tools = []
        seen_tools = set()
        for tool_name, query_text, similarity in query_matches:
            if tool_name not in seen_tools:
                selected_tools.append(tool_name)
                seen_tools.add(tool_name)
                logger.debug(f"Selected '{tool_name}' (similarity: {similarity:.3f}, matched query: '{query_text}')")
                
                if len(selected_tools) >= max_tools:
                    break
        
        return selected_tools

