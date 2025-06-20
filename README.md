# Medicaid Chatbot: Conceptual Architecture Framework

A conceptual framework demonstrating architectural thinking for pdf document based chatbot system. Showcases integration of LangChain, MCP, and LLM services in a modular RAG architecture.

## Framework Integration

Integration Architecture:

- LangChain Framework Layer
  - Document Processing: PDF extraction and text manipulation
  - RAG Chains: Composable retrieval-augmented generation workflows
  - Vector Stores: Semantic search infrastructure abstractions

- MCP Protocol Layer
  - Tool Discovery: Standardized capability advertisement
  - Resource Management: Uniform document access patterns

- LLM Service Layer
  - Prompt Engineering: Model-specific formatting and optimization
  - Context Management: Intelligent information aggregation
  - Generation Strategy: Provider-agnostic response synthesis

- Orchestration Layer
  - Component Coordination: Inter-framework communication
  - Protocol Translation: Format adaptation between systems
  - State Management: Conversation and workflow persistence
  - Workflow Execution: Multi-step process coordination

## Data Flow Pipeline

1. Input Processing: PDF Document → Text Extraction → Semantic Chunking
2. Semantic Processing: Vector Embedding → Similarity Search → Context Retrieval
3. Generation: Prompt Engineering → LLM Generation → Response Synthesis
4. Memory: Conversation Memory → User Response

## Project Structure

```
medicaid-chatbot/
├── src/
│   ├── config/
│   │   └── settings.py              # Environment-driven configuration patterns
│   ├── core/                        # RAG pipeline components
│   │   ├── document_processor.py    # PDF → structured data transformation
│   │   ├── retrieval_engine.py      # Semantic search orchestration
│   │   ├── response_generator.py    # Context-aware text generation
│   │   ├── conversation_memory.py   # Multi-turn state management
│   │   └── model_adaptation.py      # PEFT fine-tuning capabilities
│   └── agents/
│       └── orchestrator.py          # Multi-agent workflow coordination
├── tests/                           # End-to-end testing framework
│   └── e2e/                         # Complete workflow validation
├── data/
│   └── ABHIL_Member_Handbook.pdf   # Domain-specific knowledge source
├── Dockerfile                       # Multi-stage containerization
├── docker-compose.yml              # Complete system orchestration
├── requirements.txt                 # Framework dependencies
└── ARCHITECTURAL_CONCEPTS.md        # Detailed design documentation
```

## Core Components

### settings.py
Separates LLM, Vector, and MCP configurations. MCP capabilities include tools (extract_pdf_text, chunk_document, semantic_search, generate_response) and resources (pdf_documents, text_chunks, conversation_history, vector_embeddings).

### document_processor.py
Strategy pattern for PDF processing. PDFExtractor abstraction supports multiple libraries. DocumentChunk encapsulates content + metadata + embeddings. TextChunker balances context preservation with retrieval precision.

### retrieval_engine.py
Pattern for semantic search. EmbeddingModel abstracts text-to-vector transformation. VectorDatabase abstracts similarity search. Includes caching and component swapping capabilities.

### response_generator.py
Template Method pattern for response generation. PromptLibrary manages templates. LLMInterface provides provider-agnostic model access. Handles context preparation and prompt engineering.

### orchestrator.py
Agent pattern for workflow coordination. Specialized agents: QueryAnalysisAgent, RetrievalAgent, ResponseAgent. Workflow engine manages dependencies and execution order. Standardized inter-agent communication.

### conversation_memory.py
Conversation management. Turn-based persistence, entity tracking, and state management (active, waiting, completed).

### model_adaptation.py
PEFT fine-tuning patterns. Strategy pattern for LoRA, QLoRA, Adapter methods. Updates 0.1-1% of parameters for domain specialization without full retraining.

## Testing Architecture

### End-to-End Tests
Complete system validation concepts. Tests user workflow patterns, system quality metrics, acceptance criteria, and compliance validation in an abstract framework.

## Deployment Configuration

### Dockerfile
Multi-stage containerization concept. Demonstrates dependency isolation, environment setup, and specialized build targets.

### docker-compose.yml
Service orchestration concept. Shows core application, data persistence, performance layers, and environment-specific configurations.

