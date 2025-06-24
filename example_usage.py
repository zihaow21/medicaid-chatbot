#!/usr/bin/env python3
"""
Multi-Model Usage Example
Demonstrates switching between Llama 3.2 and OpenAI models
"""

import os
from src.core.response_generator import LLMFactory
from src.config.settings import LLMConfig

def demonstrate_models():
    """Demonstrate both Llama 3.2 and OpenAI model usage"""
    
    print("Multi-Model Framework Demo")
    print("=" * 40)
    
    # Example 1: Llama 3.2 (Local)
    print("\nLlama 3.2 Local Model")
    llama_config = LLMConfig(
        model_name="llama-3.2",
        provider="local",
        temperature=0.7
    )
    llama_model = LLMFactory.create_model(llama_config)
    print(f"Model Info: {llama_model.get_model_info()}")
    print(f"Response: {llama_model.generate('What is Medicaid?')}")
    
    # Example 2: OpenAI GPT (API-based)
    print("\nOpenAI GPT Model")
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai_config = LLMConfig(
            model_name="gpt-3.5-turbo",
            provider="openai",
            temperature=0.7,
            api_key=api_key
        )
        try:
            openai_model = LLMFactory.create_model(openai_config)
            print(f"Model Info: {openai_model.get_model_info()}")
            print(f"Response: {openai_model.generate('What is Medicaid?')}")
        except ValueError as e:
            print(f"OpenAI Error: {e}")
    else:
        print("OPENAI_API_KEY not found in environment")
        print("Set your API key: export OPENAI_API_KEY='your-key-here'")


if __name__ == "__main__":
    demonstrate_models() 