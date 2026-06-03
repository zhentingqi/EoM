"""
LLM client utilities.

This file defines the shared LLM client interface plus the supported backend
implementations used by HayekMAS runtimes and adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
import os
import time
import json

from hayekmas.utils.logger import logger


# =============================================================================
# SHARED LLM CONFIG
# =============================================================================

@dataclass
class LLMConfig:
    """Framework-level configuration for constructing an ``LLMClient``.

    ``api`` (backend choice) and ``name`` (model) are required. ``api_key``
    is optional at the config level because some backends (notably
    ``localhost``) do not need one; backends that do require a key enforce
    its presence at their own constructor.
    """

    api: str
    name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# BASE CLIENT
# =============================================================================

class LLMClient(ABC):
    """Abstract base class for LLM API clients.

    Args:
        model: Model identifier understood by the backend.
        max_tokens: Default token budget for generation.
        temperature: Default sampling temperature.
        **kwargs: Backend-specific defaults kept by subclasses.
    """
    
    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ):
        self.model = model
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature

    @property
    @abstractmethod
    def api_name(self) -> str:
        """Return the API backend name."""
        pass
    
    MAX_EMPTY_RETRIES = 3

    @abstractmethod
    def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Backend-specific generation. Subclasses must implement this.

        Args:
            prompt: Prompt text to send to the model.
            system_prompt: Optional system instruction prepended by the client.
            max_tokens: Optional per-call max token override.
            temperature: Optional per-call sampling temperature override.
            stop: Optional list of stop strings.
            **kwargs: Backend-specific generation arguments.

        Returns:
            The generated text response.
        """
        pass

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate a response, retrying on empty results.

        Delegates to the backend-specific ``_generate_impl`` and retries up to
        ``MAX_EMPTY_RETRIES`` times when the model returns an empty string.
        """
        for attempt in range(self.MAX_EMPTY_RETRIES):
            result = self._generate_impl(
                prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                **kwargs,
            )
            if result and result.strip():
                return result
            logger.log(
                f"[LLM] WARNING: empty response from {self.api_name} "
                f"(attempt {attempt + 1}/{self.MAX_EMPTY_RETRIES})"
            )
        return result

    def as_callable(
        self,
        system_prompt: Optional[str] = None,
        **default_kwargs
    ) -> Callable[[str], str]:
        """Return a simple callable wrapper for agent use.

        Args:
            system_prompt: Optional fixed system prompt to bind.
            **default_kwargs: Default generation kwargs applied to each call.

        Returns:
            A callable that accepts a prompt string and returns generated text.
        """
        def llm_fn(prompt: str) -> str:
            return self.generate(prompt, system_prompt=system_prompt, **default_kwargs)
        return llm_fn
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


# =============================================================================
# VLLM CLIENT
# =============================================================================

class VLLMClient(LLMClient):
    """
    vLLM client for fast local inference.
    
    Requires: pip install vllm
    """
    
    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.7,
        seed: int = 42,
        **kwargs
    ):
        super().__init__(model, max_tokens, temperature, **kwargs)
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.seed = seed
        self._llm = None
    
    @property
    def api_name(self) -> str:
        return "vllm"
    
    def _load_model(self):
        """Lazy load the vLLM model."""
        if self._llm is None:
            try:
                from vllm import LLM
            except ImportError:
                raise ImportError("vllm not installed. Run: pip install vllm")
            
            logger.log(f"[LLM] Loading vLLM: {self.model}")
            self._llm = LLM(
                model=self.model,
                max_model_len=self.max_model_len,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                seed=self.seed,
                enable_prefix_caching=True,
            )
    
    def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        self._load_model()
        from vllm import SamplingParams
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        params = SamplingParams(
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature or self.default_temperature,
            stop=stop or [],
            **kwargs
        )
        
        outputs = self._llm.generate([full_prompt], params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()
    


# =============================================================================
# SGLANG CLIENT
# =============================================================================

class SGLangClient(LLMClient):
    """
    SGLang client for fast local inference.
    
    Requires: pip install sglang
    """
    
    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ):
        super().__init__(model, max_tokens, temperature, **kwargs)
        self._engine = None
    
    @property
    def api_name(self) -> str:
        return "sglang"
    
    def _load_model(self):
        """Lazy load the SGLang engine."""
        if self._engine is None:
            try:
                import sglang as sgl
            except ImportError:
                raise ImportError("sglang not installed. Run: pip install sglang")
            
            logger.log(f"[LLM] Loading SGLang: {self.model}")
            self._engine = sgl.Engine(model_path=self.model, device='cuda')
    
    def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        self._load_model()
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        gen_kwargs = {
            "n": 1,
            "max_new_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature or self.default_temperature,
            "stop": stop or [],
            "skip_special_tokens": False,
        }
        
        outputs = self._engine.generate([full_prompt], gen_kwargs)
        return outputs[0]["text"].strip()
    


# =============================================================================
# LITELLM CLIENT
# =============================================================================

class LiteLLMClient(LLMClient):
    """
    LiteLLM client for unified access to cloud APIs.
    
    Supports: OpenAI, Anthropic, Together, Cohere, etc.
    
    Model format: "provider/model" (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
    
    Requires: pip install litellm
    """
    
    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        max_retries: int = 5,
        base_delay: float = 1.0,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        force_openai_routing: Optional[bool] = None,
        drop_params: bool = True,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream: Optional[bool] = None,
        **kwargs
    ):
        super().__init__(model, max_tokens, temperature, **kwargs)
        self.max_retries = max_retries
        self.base_delay = base_delay
        # OpenAI-compatible routing support (e.g., custom gateways exposing GPT/Gemini).
        self.api_base = api_base or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if force_openai_routing is None:
            env_toggle = os.environ.get("LITELLM_FORCE_OPENAI_ROUTING")
            if env_toggle is None:
                self.force_openai_routing = True
            else:
                self.force_openai_routing = env_toggle.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.force_openai_routing = force_openai_routing
        self.drop_params = drop_params
        self.top_p = top_p
        self.top_k = top_k
        self.stream = stream
        self.extra_kwargs = kwargs
    
    @property
    def api_name(self) -> str:
        return "litellm"

    def _resolve_model_name(self) -> str:
        """
        Resolve LiteLLM model name.

        When force_openai_routing is enabled, bare names like 'gpt-5-nano' or
        'gemini-3-flash' are normalized to 'openai/<name>' for OpenAI-compatible gateways.
        """
        model_name = self.model.strip()
        if self.force_openai_routing and "/" not in model_name:
            return f"openai/{model_name}"
        return model_name

    def _max_token_fallbacks(self, requested_max_tokens: Optional[int]) -> List[int]:
        """Return a descending list of max_tokens candidates to try."""
        primary = requested_max_tokens or self.default_max_tokens
        if "gemini" in self.model.lower():
            primary = min(primary, 65536)
        candidates = [primary]
        for fallback in (32768, 16384, 8192, 4096, 2048):
            if fallback < primary and fallback not in candidates:
                candidates.append(fallback)
        return candidates

    def _use_streaming(self, stream_override: Optional[bool]) -> bool:
        """Return whether this request should use streaming mode."""
        if stream_override is not None:
            return stream_override
        if self.stream is not None:
            return self.stream
        return False

    def _read_streaming_content(self, response) -> str:
        """Collect text content from a LiteLLM streaming response."""
        parts: list[str] = []
        for chunk in response:
            try:
                choices = getattr(chunk, "choices", None)
                if choices is None and isinstance(chunk, dict):
                    choices = chunk.get("choices")
                if not choices:
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None and isinstance(choice, dict):
                    delta = choice.get("delta") or choice.get("message")
                content = getattr(delta, "content", None) if delta is not None else None
                if content is None and isinstance(delta, dict):
                    content = delta.get("content")
                if isinstance(content, str) and content:
                    parts.append(content)
                elif isinstance(content, list):
                    parts.extend(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
            except Exception:
                continue
        return "".join(parts).strip()
    
    def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        try:
            from litellm import completion
            from litellm.exceptions import BadRequestError, ServiceUnavailableError, RateLimitError
        except ImportError:
            raise ImportError("litellm not installed. Run: pip install litellm")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        stream_override = kwargs.pop("stream", None)
        use_stream = self._use_streaming(stream_override)
        max_token_candidates = self._max_token_fallbacks(max_tokens)
        for token_idx, max_token_candidate in enumerate(max_token_candidates):
            for attempt in range(self.max_retries):
                try:
                    request_extras = dict(self.extra_kwargs)
                    request_extras.update(kwargs)
                    request_kwargs: Dict[str, Any] = {
                        "model": self._resolve_model_name(),
                        "messages": messages,
                        "max_tokens": max_token_candidate,
                        "temperature": temperature or self.default_temperature,
                        "stop": stop,
                        "stream": use_stream,
                        "drop_params": self.drop_params,
                        # Bound the per-request wall-clock so a stuck connection
                        # (rate-limit hang, server stall, dropped TCP) cannot
                        # freeze a parallel run for hours. A normal call
                        # finishes in 5-30 s; 180 s is generous.
                        "timeout": request_extras.pop("timeout", 180),
                        **request_extras,
                    }
                    # Only pass top_p / top_k when explicitly provided by user
                    top_p_val = request_extras.pop("top_p", self.top_p)
                    if top_p_val is not None:
                        request_kwargs["top_p"] = top_p_val
                    top_k_val = request_extras.pop("top_k", self.top_k)
                    if top_k_val is not None:
                        request_kwargs["top_k"] = top_k_val
                    if self.api_base:
                        request_kwargs["api_base"] = self.api_base
                    if self.api_key:
                        request_kwargs["api_key"] = self.api_key

                    response = completion(**request_kwargs)
                    if use_stream:
                        return self._read_streaming_content(response)
                    choice = response.choices[0]
                    response_text = (choice.message.content or "").strip()
                    if not response_text:
                        finish = getattr(choice, "finish_reason", None)
                        logger.log(f"[LLM] empty content (finish_reason={finish})")
                    return response_text

                except BadRequestError as e:
                    message = str(e)
                    lower_message = message.lower()
                    is_invalid_argument = (
                        "invalid argument" in lower_message
                        or "maxoutputtokens" in lower_message
                        or "supported range" in lower_message
                    )
                    has_smaller_candidate = token_idx < len(max_token_candidates) - 1
                    if is_invalid_argument and has_smaller_candidate:
                        next_tokens = max_token_candidates[token_idx + 1]
                        logger.log(
                            f"[LLM] BadRequest for max_tokens={max_token_candidate}; retrying with max_tokens={next_tokens}..."
                        )
                        break
                    logger.log(f"[LLM] Error: {e}")
                    raise

                except (ServiceUnavailableError, RateLimitError) as e:
                    if attempt < self.max_retries - 1:
                        delay = self.base_delay * (2 ** attempt)
                        logger.log(f"[LLM] {type(e).__name__}, retry in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        raise
                except Exception as e:
                    logger.log(f"[LLM] Error: {e}")
                    raise

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """Fallback batch generation by repeated single calls."""
        return [
            self.generate(prompt, system_prompt=system_prompt, **kwargs)
            for prompt in prompts
        ]


# =============================================================================
# LOCALHOST OPENAI-COMPATIBLE CLIENT
# =============================================================================

class LocalhostClient(LLMClient):
    """
    OpenAI-compatible localhost client (e.g., SGLang / vLLM server).

    Defaults to http://localhost:8000/v1 and auto-detects model name from
    the `/models` endpoint when model is not provided.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        top_p: float = 1.0,
        max_retries: int = 5,
        base_delay: float = 1.0,
        timeout: float = 300.0,
        **kwargs
    ):
        super().__init__(model=model, max_tokens=max_tokens, temperature=temperature, **kwargs)
        self.api_base = (api_base or os.environ.get("LOCALHOST_BASE_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LOCALHOST_API_KEY") or "not-needed"
        self.top_p = top_p
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self._model_detected = bool(model)

    @property
    def api_name(self) -> str:
        return "localhost"

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _detect_model(self) -> None:
        if self._model_detected:
            return
        try:
            import requests
        except ImportError:
            raise ImportError("requests not installed. Run: pip install requests")

        try:
            response = requests.get(
                f"{self.api_base}/models",
                headers=self._headers(),
                timeout=min(self.timeout, 30.0),
            )
            response.raise_for_status()
            data = response.json()
            models = data.get("data", []) if isinstance(data, dict) else []
            if models and isinstance(models[0], dict):
                self.model = str(models[0].get("id", "")).strip()
                if self.model:
                    logger.log(f"[LLM] Auto-detected localhost model: {self.model}")
            if not self.model:
                logger.log("[LLM] Warning: localhost /models returned no model id; using empty model name")
        except Exception as e:
            logger.log(f"[LLM] Warning: could not auto-detect localhost model: {e}")
        finally:
            self._model_detected = True

    def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        try:
            import requests
        except ImportError:
            raise ImportError("requests not installed. Run: pip install requests")

        self._detect_model()

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature or self.default_temperature,
            "top_p": kwargs.pop("top_p", self.top_p),
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt < self.max_retries - 1:
                        delay = self.base_delay * (2 ** attempt)
                        logger.log(
                            f"[LLM] localhost {response.status_code}, retry in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        continue
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    return ""
                message = choices[0].get("message") or {}
                content = message.get("content")
                reasoning_content = message.get("reasoning_content")
                if content is None and reasoning_content:
                    content = reasoning_content
                if content is None:
                    content = "Empty response."
                if isinstance(content, list):
                    return "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    ).strip()
                return str(content).strip()
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.log(f"[LLM] localhost error: {e}, retry in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise


# =============================================================================
# TOGETHER CLIENT
# =============================================================================

class TogetherClient(LLMClient):
    """
    Together AI chat-completions client.

    Supports Together's native Kimi K2.5 thinking mode by default when the
    selected model name contains `Kimi-K2.5`.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        api_url: str = "https://api.together.xyz/v1/chat/completions",
        max_retries: int = 5,
        reasoning_enabled: bool = True,
        top_p: float = 0.95,
        stream: Optional[bool] = None,
        **kwargs
    ):
        super().__init__(model, max_tokens, temperature, **kwargs)
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together API key required. Set TOGETHER_API_KEY env var or pass api_key.")
        self.api_url = api_url
        self.max_retries = max_retries
        self.reasoning_enabled = reasoning_enabled
        self.top_p = top_p
        self.stream = stream

    @property
    def api_name(self) -> str:
        return "together"

    def _is_kimi_thinking_model(self) -> bool:
        return "kimi-k2.5" in self.model.lower()

    def _resolve_max_tokens(self, requested_max_tokens: Optional[int]) -> int:
        """
        Kimi K2.5 thinking mode can spend a large fraction of the output budget on
        reasoning tokens before emitting final content. Use a much larger default
        budget unless the caller explicitly overrides it.
        """
        if requested_max_tokens is not None:
            return requested_max_tokens
        if self._is_kimi_thinking_model() and self.default_max_tokens == 2048:
            return 96000
        return self.default_max_tokens

    def _use_streaming(self, stream_override: Optional[bool]) -> bool:
        if stream_override is not None:
            return stream_override
        if self.stream is not None:
            return self.stream
        return self._is_kimi_thinking_model()

    def _read_streaming_content(self, response) -> str:
        collected_content: list[str] = []
        collected_reasoning: list[str] = []
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            choices = payload.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content")
            reasoning = delta.get("reasoning")
            if isinstance(content, str) and content:
                collected_content.append(content)
            elif isinstance(content, list):
                collected_content.extend(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if isinstance(reasoning, str) and reasoning:
                collected_reasoning.append(reasoning)

        text = "".join(collected_content).strip()
        if text:
            return text
        return ""

    def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        try:
            import requests
        except ImportError:
            raise ImportError("requests not installed. Run: pip install requests")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        stream_override = kwargs.pop("stream", None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resolved_temperature = (
            temperature
            if temperature is not None
            else (1.0 if self._is_kimi_thinking_model() else self.default_temperature)
        )

        use_stream = self._use_streaming(stream_override)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self._resolve_max_tokens(max_tokens),
            "temperature": resolved_temperature,
            "top_p": kwargs.pop("top_p", self.top_p),
            "stream": use_stream,
        }
        if stop:
            payload["stop"] = stop
        if self.reasoning_enabled and self._is_kimi_thinking_model():
            payload["reasoning"] = kwargs.pop("reasoning", {"enabled": True})
        payload.update(kwargs)

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=180,
                    stream=use_stream,
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt < self.max_retries - 1:
                        delay = 2 ** attempt
                        logger.log(f"[LLM] Together {response.status_code}, retry in {delay}s...")
                        time.sleep(delay)
                        continue

                response.raise_for_status()
                if use_stream:
                    text = self._read_streaming_content(response)
                    reasoning = ""
                else:
                    data = response.json()
                    choices = data.get("choices") or []
                    if not choices:
                        return ""
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    if isinstance(content, list):
                        text = "".join(
                            part.get("text", "") for part in content if isinstance(part, dict)
                        ).strip()
                    else:
                        text = str(content).strip()
                    reasoning = str(message.get("reasoning", "")).strip()

                if text:
                    return text

                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt
                    logger.log(
                        f"[LLM] Together returned empty content (reasoning_len={len(reasoning)}), retry in {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                return text
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt
                    logger.log(f"[LLM] Together error: {e}, retry in {delay}s...")
                    time.sleep(delay)
                else:
                    raise


# =============================================================================
# DEMO CLIENT
# =============================================================================

class DemoClient(LLMClient):
    """
    Demo client for testing without real LLM calls.
    
    Evaluates simple arithmetic expressions.
    """
    
    def __init__(self, **kwargs):
        super().__init__(model="demo", **kwargs)
    
    @property
    def api_name(self) -> str:
        return "demo"
    
    def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Evaluate arithmetic expressions. For testing only!"""
        try:
            if "Problem:" in prompt:
                expr = prompt.split("Problem:", 1)[1].strip().splitlines()[0]
            else:
                expr = prompt.strip().splitlines()[-1]
            
            result = eval(expr, {"__builtins__": {}})
            return str(result)
        except Exception:
            return ""


# =============================================================================
# FACTORY
# =============================================================================

_API_CLASSES = {
    "vllm": VLLMClient,
    "sglang": SGLangClient,
    "litellm": LiteLLMClient,
    "localhost": LocalhostClient,
    "together": TogetherClient,
    "demo": DemoClient,
}


def get_available_apis() -> List[str]:
    """Return the list of available API backend names."""
    return list(_API_CLASSES.keys())


def get_llm_client(api: str, model: str, **kwargs) -> LLMClient:
    """
    Create an LLM client for the requested backend.

    Args:
        api: Backend name such as `vllm`, `sglang`, `litellm`, `together`,
            `localhost`, or `demo`.
        model: Model name or model path understood by the backend.
        **kwargs: Backend-specific constructor parameters.

    Returns:
        A configured `LLMClient` subclass instance.

    Raises:
        ValueError: If `api` is unknown.
    """
    if api not in _API_CLASSES:
        raise ValueError(f"Unknown API: {api}. Available: {get_available_apis()}")
    
    return _API_CLASSES[api](model=model, **kwargs)
