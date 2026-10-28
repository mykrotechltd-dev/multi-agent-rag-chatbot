"""Base agent class for specialist agents using LangChain LCEL chains."""

import os
from abc import ABC
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse import observe
from langfuse.langchain import CallbackHandler

# Load environment variables
load_dotenv()

from typing import Optional, Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

from config import LLM_MODEL, MIN_SIMILARITY, DEFAULT_K
from indexing.embeddings import generate_embedding_for_text, load_vector_store
from querying.tools.rag_tool import get_rag_tools_for_agent
from querying.tools.qdrant_source_store import get_qdrant_source_store
from utils.llm import initialize_llm
from utils.langfuse_config import configure_langfuse_environment


@dataclass
class AgentResponse:
    """Response from an agent."""
    content: str
    agent_name: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for specialist agents using LCEL chains."""
    
    def __init__(
        self,
        name: str,
        handbook_name: str,
        description: str,
        llm_model: str = None,
        vector_store: Optional[Union[Chroma, FAISS]] = None,
    ):
        """
        Initialize the agent.
        
        Args:
            name: Agent name (e.g., "finance", "hr")
            handbook_name: Name of the handbook/vector store
            description: Agent description
            llm_model: LLM model to use. Defaults to config LLM_MODEL.
            vector_store: Preloaded vector store. If None, will load on demand (slower)
        """
        self.name = name
        self.handbook_name = handbook_name
        self.description = description
        self.llm_model = llm_model or LLM_MODEL
        
        # Initialize Langfuse callback handler
        # CallbackHandler reads from environment variables automatically
        configure_langfuse_environment()
        self.langfuse_handler = CallbackHandler()
        
        # Initialize LLM
        self._initialize_llm()
        
        # Store preloaded vector store or None for lazy loading
        self._vector_store = vector_store
        
        # Get RAG tool for this agent (pass preloaded vector store)
        self.rag_tool = get_rag_tools_for_agent(name, handbook_name, vector_store=vector_store)[0]
        
        # Create LCEL chain that always uses RAG tool first
        self._create_rag_chain()
    
    def _initialize_llm(self):
        """Initialize the LLM with Langfuse instrumentation."""
        self.llm = initialize_llm(
            model=self.llm_model,
            langfuse_handler=self.langfuse_handler
        )
    
    def _create_rag_chain(self):
        """Create LCEL chain that always uses RAG tool first, then formats response."""
        no_info_response = f"I don't have information about this in the {self.handbook_name} knowledge base. This question may be outside my area of expertise. Please try rephrasing your question or contact support for assistance."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a {self.name.upper()} specialist agent for JupiterIQ, a SaaS company.

Your role: {self.description}

You have access to tools to search the {self.handbook_name} knowledge base. Always use the {self.name}_rag_search tool to search the knowledge base when answering questions.

Guidelines:
1. Always use the {self.name}_rag_search tool to search the knowledge base when answering questions
2. Answer based on the retrieved context from the tool
3. If the context doesn't contain enough information, say so clearly
4. Be concise but thorough
5. Maintain a professional, helpful tone
6. If the query is outside your domain, acknowledge it and suggest routing to another agent"""),
            ("human", f"""Previous conversation:
{{conversation_history}}

User question: {{query}}

Retrieved context from {self.handbook_name} knowledge base:
{{context}}

Answer the user's question based ONLY on the retrieved context above. If the context says "No relevant information found in {self.handbook_name} knowledge base", you must respond: "{no_info_response}" """)
        ])
        
        # Create LCEL chain: prompt -> LLM -> output parser
        self.rag_chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _load_vector_store(self):
        """Lazy load the vector store."""
        if self._vector_store is None:
            self._vector_store = load_vector_store(self.handbook_name)
        return self._vector_store
    
    def _retrieve_context(
        self, 
        query: str, 
        k: int = DEFAULT_K,
        min_similarity: float = MIN_SIMILARITY
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from the vector store.
        
        Uses same logic as RAG tool: retrieves k*2, filters by similarity, 
        deduplicates, returns top k. Ensures sources match what LLM uses.
        
        Chroma returns cosine distance (0 = identical, 2 = opposite).
        Qdrant returns cosine similarity directly. Both use the same threshold.
        
        Args:
            query: User query
            k: Number of documents to retrieve (final count after filtering)
            min_similarity: Minimum similarity threshold (0.0 to 1.0). 
                          Defaults to config MIN_SIMILARITY.
            
        Returns:
            List of retrieved documents with metadata, filtered and deduplicated
        """
        
        vector_store = self._load_vector_store()
        query_vector = generate_embedding_for_text(query)
        candidates: list[Dict[str, Any]] = []

        # Reuse one embedding for the bundled handbook and uploaded Qdrant sources.
        docs = vector_store.similarity_search_by_vector_with_score(query_vector, k=k * 2)
        for doc, distance in docs:
            similarity = 1.0 - float(distance)
            if similarity >= min_similarity:
                candidates.append(
                    {
                        "content": doc.page_content,
                        "metadata": dict(doc.metadata),
                        "similarity": similarity,
                        "distance": float(distance),
                    }
                )

        if os.getenv("QDRANT_URL"):
            try:
                uploaded_sources = get_qdrant_source_store().search(
                    query_vector=query_vector,
                    agent_name=self.name,
                    limit=k * 2,
                )
                for source in uploaded_sources:
                    if source.similarity >= min_similarity:
                        candidates.append(
                            {
                                "content": source.content,
                                "metadata": source.metadata,
                                "similarity": source.similarity,
                                "distance": source.distance,
                            }
                        )
            except Exception as exc:
                # The bundled handbooks remain useful while the external vector service is unavailable.
                print(f"Warning: Qdrant source retrieval failed for {self.name}: {exc}")

        deduplicated: dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            normalized = candidate["content"].strip()
            if not normalized:
                continue
            previous = deduplicated.get(normalized)
            if previous is None or candidate["similarity"] > previous["similarity"]:
                deduplicated[normalized] = candidate

        context_docs = sorted(
            deduplicated.values(),
            key=lambda item: item["similarity"],
            reverse=True,
        )
        return context_docs[:k]
    
    @observe(name="agent_process_query")
    def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        k: int = DEFAULT_K,
        min_similarity: float = MIN_SIMILARITY,
    ) -> AgentResponse:
        """
        Process a query and generate a response using LCEL chain.
        
        Args:
            query: User query
            conversation_history: Previous conversation messages [{"role": "user/assistant", "content": "..."}]
            k: Number of documents to retrieve
            min_similarity: Minimum similarity threshold (0.0 to 1.0). Defaults to config MIN_SIMILARITY.
            
        Returns:
            AgentResponse with answer and sources
        """
        # Metadata is captured automatically by @observe decorator
        
        try:
            # Format conversation history if provided
            if conversation_history:
                history_context = "\n".join([
                    f"{msg['role'].title()}: {msg['content']}"
                    for msg in conversation_history[-5:]  # Last 5 messages
                ])
            else:
                history_context = "None"
            
            # Single retrieval call - get documents once
            context_docs = self._retrieve_context(query, k=k, min_similarity=min_similarity)
            
            # Format context for LLM (same format as RAG tool)
            if not context_docs:
                retrieved_context = f"No relevant information found in {self.handbook_name} knowledge base."
            else:
                context_parts = []
                for i, doc in enumerate(context_docs, 1):
                    context_parts.append(
                        f"[Source {i}] (Similarity: {doc['similarity']:.2f})\n{doc['content']}"
                    )
                retrieved_context = "\n\n".join(context_parts)
            
            # Use same documents for sources
            sources = [
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity": doc["similarity"],
                    "distance": doc.get("distance"),
                }
                for doc in context_docs
            ]
            
            # Run LCEL chain with retrieved context
            response_content = self.rag_chain.invoke(
                {
                    "query": query,
                    "context": retrieved_context,
                    "conversation_history": history_context,
                },
                config={"callbacks": [self.langfuse_handler]}
            )
            
            # Metadata captured by @observe decorator
            
            return AgentResponse(
                content=response_content,
                agent_name=self.name,
                sources=sources,
                metadata={"success": True},
            )
            
        except Exception as e:
            # Error is automatically captured by @observe decorator
            return AgentResponse(
                content=f"I encountered an error while processing your query. Please try again or contact support if the issue persists.",
                agent_name=self.name,
                sources=[],
                metadata={"error": str(e), "error_type": type(e).__name__},
            )
