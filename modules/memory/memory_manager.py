"""Memory manager for storing and retrieving memories."""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import faiss  # type: ignore
import numpy as np

from .embedding_service import EmbeddingService
from .models import Memory, MemoryQuestion, MemorySearchResult


class MemoryManager:
    """Manager for storing and retrieving memories with embeddings."""

    def __init__(
        self,
        llm: Any = None,
        memory_db_path: str = "memory.db",
        embedding_model: str = "ibm-granite/granite-embedding-english-r2",
        num_questions_per_memory: int = 2,
        cache_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize the memory manager."""
        self.llm = llm
        self.memory_db_path = memory_db_path
        self.num_questions_per_memory = num_questions_per_memory
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger or logging.getLogger("ymca.memory")

        self.embedding_service = EmbeddingService(embedding_model, cache_dir, logger)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.question_id_map: List[str] = []

        self._init_database()
        self._load_index()

    def _init_database(self) -> None:
        """Initialize the SQLite database."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        tags TEXT NOT NULL,
                        url TEXT,
                        line_numbers TEXT
                    )
                """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS questions (
                        id TEXT PRIMARY KEY,
                        memory_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        FOREIGN KEY (memory_id) REFERENCES memories (id) ON DELETE CASCADE
                    )
                """
                )

                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise

    def _load_index(self) -> None:
        """Load or create the FAISS index from stored questions."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.execute("SELECT id, embedding_json FROM questions")
                questions_data = cursor.fetchall()

            if not questions_data:
                return

            embeddings = []
            question_ids = []

            for question_id, embedding_json in questions_data:
                try:
                    embedding = json.loads(embedding_json)
                    embeddings.append(embedding)
                    question_ids.append(question_id)
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to decode embedding for question {question_id}")

            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                dimension = embeddings_array.shape[1]

                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(embeddings_array)
                self.question_id_map = question_ids

        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")

    def _generate_summary_and_questions(self, content: str) -> tuple[str, List[str]]:
        """Generate summary and questions for memory content."""
        if not self.llm:
            raise RuntimeError("No LLM available for generating summaries and questions")

        summary_prompt = f"""Please create a concise summary of the following text in 1-7 sentences. Provide only one summary, not multiple versions.

{content}

Summary:"""
        summary_response = self.llm.generate_response(summary_prompt).strip()

        # Extract only the first meaningful summary line
        summary_lines = []
        for line in summary_response.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Skip lines that are prompts, headers, or meta-commentary
            skip_patterns = ["Summary:", "Please create", "**Note:**", "Note:", "This revised summary", "This summary", "The summary", "maintaining the", "condenses the", "refutation of"]

            if any(line.startswith(pattern) or pattern.lower() in line.lower() for pattern in skip_patterns):
                continue

            # Skip lines with meta-commentary words
            meta_words = ["generate", "create", "provide", "revised", "condenses", "maintaining", "refutation", "misconception"]
            if any(word in line.lower() for word in meta_words):
                continue

            # Take the first meaningful content line as our summary
            if line and len(line) > 10:  # Ensure it's substantial
                summary_lines.append(line)
                break

        summary = summary_lines[0] if summary_lines else summary_response.split("\n")[0].strip()

        questions_prompt = f"""Based on the following text, generate exactly {self.num_questions_per_memory} specific questions that this text can answer. Each question should be on a separate line and be concise.

{content}

Questions:"""
        questions_response = self.llm.generate_response(questions_prompt).strip()

        questions = []
        for line in questions_response.split("\n"):
            line = line.strip()
            if line and not line.startswith("Questions:"):
                line = line.lstrip("0123456789.- ")
                if line:
                    questions.append(line)

        questions = questions[: self.num_questions_per_memory]
        if len(questions) < self.num_questions_per_memory:
            raise RuntimeError(f"LLM generated only {len(questions)} questions, expected {self.num_questions_per_memory}")

        return summary, questions

    def _split_content_into_chunks(self, content: str) -> List[str]:
        """Split content into chunks with overlap if it exceeds max_chunk_size."""
        if len(content) <= self.max_chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            # Calculate the end position for this chunk
            end = start + self.max_chunk_size

            # If this is not the last chunk, try to find a good breaking point
            if end < len(content):
                # Look for sentence boundaries (. ! ?) within the last 200 characters
                search_start = max(start + self.max_chunk_size - 200, start)
                search_text = content[search_start:end]

                # Find the last sentence boundary
                for delimiter in [". ", "! ", "? ", "\n\n"]:
                    last_pos = search_text.rfind(delimiter)
                    if last_pos > 0:
                        # Adjust end to include the delimiter
                        end = search_start + last_pos + len(delimiter)
                        break

            # Extract the chunk
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # If this was the last chunk, break
            if end >= len(content):
                break

            # Calculate next start position with overlap
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def store_memory(self, content: str, tags: Optional[List[str]] = None, url: Optional[str] = None, line_numbers: Optional[str] = None) -> str:
        """Store a new memory with optional source tracking.

        Args:
            content: The memory content
            tags: Optional tags for the memory
            url: Optional URL where the content came from
            line_numbers: Optional string representing line numbers (e.g., "10-15", "10", "10,12-15")

        Returns:
            The memory ID (or the first chunk's ID if chunked)
        """
        if tags is None:
            tags = []

        # Split content into chunks if it's too large
        chunks = self._split_content_into_chunks(content)

        if len(chunks) == 1:
            # Single chunk, store normally
            return self._store_single_memory(content, tags, url, line_numbers)
        else:
            # Multiple chunks, store each as a separate memory with chunk tags
            self.logger.info(f"Splitting large memory ({len(content)} chars) into {len(chunks)} chunks")

            chunk_ids = []

            for i, chunk in enumerate(chunks):
                # Add chunk-specific tags
                chunk_tags = tags.copy()
                chunk_tags.extend([f"chunk_{i+1}_of_{len(chunks)}", "chunked_memory"])

                # For the first chunk, also add original_chunk tag
                if i == 0:
                    chunk_tags.append("original_chunk")

                chunk_id = self._store_single_memory(chunk, chunk_tags, url, line_numbers)
                chunk_ids.append(chunk_id)

            self.logger.info(f"Stored {len(chunks)} memory chunks with IDs: {chunk_ids}")
            # Return the first chunk's ID as the primary memory ID
            return chunk_ids[0]

    def _store_single_memory(self, content: str, tags: Optional[List[str]] = None, url: Optional[str] = None, line_numbers: Optional[str] = None) -> str:
        """Store a single memory (used internally for both regular and chunked memories)."""
        if tags is None:
            tags = []

        summary, questions_text = self._generate_summary_and_questions(content)

        memory = Memory(content=content, summary=summary, tags=tags, url=url, line_numbers=line_numbers)

        for question_text in questions_text:
            question = MemoryQuestion(text=question_text, memory_id=memory.id)

            embedding = self.embedding_service.encode_single(question_text)
            question.embedding = embedding
            memory.questions.append(question)

        self._store_memory_in_db(memory)
        self._update_index_with_memory(memory)

        return memory.id

    def _store_memory_in_db(self, memory: Memory) -> None:
        """Store memory and its questions in the database."""
        with sqlite3.connect(self.memory_db_path) as conn:
            conn.execute(
                """
                INSERT INTO memories (id, content, summary, created_at, updated_at, tags, url, line_numbers)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (memory.id, memory.content, memory.summary, memory.created_at.isoformat(), memory.updated_at.isoformat(), json.dumps(memory.tags), memory.url, memory.line_numbers),
            )

            for question in memory.questions:
                conn.execute(
                    """
                    INSERT INTO questions (id, memory_id, text, embedding_json)
                    VALUES (?, ?, ?, ?)
                """,
                    (question.id, question.memory_id, question.text, json.dumps(question.embedding)),
                )

            conn.commit()

    def _update_index_with_memory(self, memory: Memory) -> None:
        """Update the FAISS index with new memory questions."""
        if not memory.questions:
            return

        embeddings = []
        question_ids = []

        for question in memory.questions:
            if question.embedding:
                embeddings.append(question.embedding)
                question_ids.append(question.id)

        if not embeddings:
            return

        embeddings_array = np.array(embeddings, dtype=np.float32)

        if self.index is None:
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.question_id_map = []

        self.index.add(embeddings_array)
        self.question_id_map.extend(question_ids)

    def search_memories(self, query: str, max_results: int = 5, similarity_threshold: float = 0.1) -> List[MemorySearchResult]:
        """Search for memories based on a query."""
        if self.index is None or len(self.question_id_map) == 0:
            return []

        try:
            query_embedding = self.embedding_service.encode_single(query)
            query_array = np.array([query_embedding], dtype=np.float32)

            scores, indices = self.index.search(query_array, min(max_results * 3, len(self.question_id_map)))

            memory_scores: Dict[str, Dict[str, Any]] = {}

            for score, idx in zip(scores[0], indices[0]):
                if score < similarity_threshold:
                    continue

                question_id = self.question_id_map[idx]

                with sqlite3.connect(self.memory_db_path) as conn:
                    cursor = conn.execute(
                        """
                        SELECT q.memory_id, q.text, m.content, m.summary, m.created_at, m.updated_at, m.tags
                        FROM questions q
                        JOIN memories m ON q.memory_id = m.id
                        WHERE q.id = ?
                    """,
                        (question_id,),
                    )

                    row = cursor.fetchone()
                    if not row:
                        continue

                    memory_id, question_text, content, summary, created_at, updated_at, tags_json = row

                    if memory_id not in memory_scores:
                        memory_scores[memory_id] = {
                            "memory": Memory(id=memory_id, content=content, summary=summary, created_at=datetime.fromisoformat(created_at), updated_at=datetime.fromisoformat(updated_at), tags=json.loads(tags_json)),
                            "best_score": float(score),
                            "matched_questions": [question_text],
                        }
                    else:
                        memory_scores[memory_id]["best_score"] = max(memory_scores[memory_id]["best_score"], float(score))
                        memory_scores[memory_id]["matched_questions"].append(question_text)

            results = []
            for memory_data in memory_scores.values():
                result = MemorySearchResult(memory=memory_data["memory"], relevance_score=memory_data["best_score"], matched_questions=memory_data["matched_questions"])
                results.append(result)

            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:max_results]

        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            return []

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT content, summary, created_at, updated_at, tags, url, line_numbers
                    FROM memories WHERE id = ?
                """,
                    (memory_id,),
                )

                memory_row = cursor.fetchone()
                if not memory_row:
                    return None

                content, summary, created_at, updated_at, tags_json, url, line_numbers = memory_row

                cursor = conn.execute(
                    """
                    SELECT id, text, embedding_json FROM questions WHERE memory_id = ?
                """,
                    (memory_id,),
                )

                questions = []
                for q_id, q_text, q_embedding_json in cursor.fetchall():
                    embedding = json.loads(q_embedding_json)
                    question = MemoryQuestion(id=q_id, text=q_text, embedding=embedding, memory_id=memory_id)
                    questions.append(question)

                memory = Memory(
                    id=memory_id,
                    content=content,
                    summary=summary,
                    questions=questions,
                    created_at=datetime.fromisoformat(created_at),
                    updated_at=datetime.fromisoformat(updated_at),
                    tags=json.loads(tags_json) if tags_json else [],
                    url=url,
                    line_numbers=line_numbers,
                )

                return memory

        except Exception as e:
            self.logger.error(f"Failed to get memory {memory_id}: {e}")
            return None

    def list_memories(self, limit: int = 50, offset: int = 0) -> List[Memory]:
        """List stored memories."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, content, summary, created_at, updated_at, tags, url, line_numbers
                    FROM memories
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                """,
                    (limit, offset),
                )

                memories = []
                for row in cursor.fetchall():
                    memory_id, content, summary, created_at, updated_at, tags_json, url, line_numbers = row

                    q_cursor = conn.execute(
                        """
                        SELECT id, text, embedding_json FROM questions WHERE memory_id = ?
                    """,
                        (memory_id,),
                    )

                    questions = []
                    for q_id, q_text, q_embedding_json in q_cursor.fetchall():
                        embedding = json.loads(q_embedding_json)
                        question = MemoryQuestion(id=q_id, text=q_text, embedding=embedding, memory_id=memory_id)
                        questions.append(question)

                    memory = Memory(
                        id=memory_id,
                        content=content,
                        summary=summary,
                        questions=questions,
                        created_at=datetime.fromisoformat(created_at),
                        updated_at=datetime.fromisoformat(updated_at),
                        tags=json.loads(tags_json) if tags_json else [],
                        url=url,
                        line_numbers=line_numbers,
                    )
                    memories.append(memory)

                return memories

        except Exception as e:
            self.logger.error(f"Failed to list memories: {e}")
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                conn.execute("DELETE FROM questions WHERE memory_id = ?", (memory_id,))
                cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

                if cursor.rowcount > 0:
                    conn.commit()
                    self._rebuild_index()
                    return True
                else:
                    return False

        except Exception as e:
            self.logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from scratch."""
        try:
            self.index = None
            self.question_id_map = []
            self._load_index()
        except Exception as e:
            self.logger.error(f"Failed to rebuild index: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM memories")
                memory_count = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM questions")
                question_count = cursor.fetchone()[0]

                cursor = conn.execute(
                    """
                    SELECT created_at FROM memories
                    ORDER BY created_at DESC LIMIT 1
                """
                )
                recent_result = cursor.fetchone()
                most_recent = recent_result[0] if recent_result else None

            embedding_stats = self.embedding_service.get_model_info()

            stats = {
                "total_memories": memory_count,
                "total_questions": question_count,
                "questions_per_memory": self.num_questions_per_memory,
                "most_recent_memory": most_recent,
                "embedding_service": embedding_stats,
                "index_size": len(self.question_id_map) if self.question_id_map else 0,
                "database_path": self.memory_db_path,
            }

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
