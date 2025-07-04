"""
Configuration Settings - Conceptual Framework

Demonstrates configuration management patterns for AI systems.
Pure architectural concepts, minimal implementation.
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class LLMConfig:
    """LLM configuration concept"""
    model_name: str = "llama-3.2"
    provider: str = "local"  # "local" for Llama, "openai" for OpenAI
    temperature: float = 0.7
    max_tokens: int = 500
    api_key: Optional[str] = None  # For API-based models like OpenAI


@dataclass
class VectorConfig:
    """Vector database configuration concept"""
    dimension: int = 384
    similarity_metric: str = "cosine"
    index_type: str = "flat"


@dataclass
class MCPConfig:
    """MCP protocol configuration concept"""
    protocol_version: str = "2024-11-05"
    server_name: str = "medicaid-chatbot"
    capabilities: list = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = ["tools", "resources"]


class Settings:
    """
    Configuration Management Pattern
    
    Concept: Environment-based configuration with defaults
    """
    
    def __init__(self):
        # Support both Llama 3.2 and OpenAI models
        model_name = os.getenv("LLM_MODEL", "llama-3.2")
        provider = "openai" if model_name.startswith("gpt-") else "local"
        
        self.llm = LLMConfig(
            model_name=model_name,
            provider=provider,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
            api_key=os.getenv("OPENAI_API_KEY")  # OpenAI API key from environment
        )
        
        self.vector = VectorConfig(
            dimension=int(os.getenv("VECTOR_DIM", "384")),
            similarity_metric=os.getenv("SIMILARITY_METRIC", "cosine")
        )
        
        self.mcp = MCPConfig(
            server_name=os.getenv("MCP_SERVER_NAME", "medicaid-chatbot")  
        )
        
        self.pdf_path = os.getenv("PDF_PATH", "data/ABHIL_Member_Handbook.pdf")


# Global settings instance
settings = Settings() 