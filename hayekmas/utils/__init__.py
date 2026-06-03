"""
Shared utilities used across the core loop and adapter runtimes.

Recommended LLM path:
    from hayekmas.utils.llm import get_llm_client
    client = get_llm_client("litellm", model="gemini-3-flash")

LLM Clients:
    from hayekmas.utils.llm import get_llm_client

    client = get_llm_client("litellm", model="openai/gpt-4")
    client = get_llm_client("vllm", model="/path/to/model")
    client = get_llm_client("sglang", model="/path/to/model")
    client = get_llm_client("localhost", model="")
"""
