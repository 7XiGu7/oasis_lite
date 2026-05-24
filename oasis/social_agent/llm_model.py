import asyncio
import datetime
import json
import logging
import os
import re
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

import tiktoken
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    DefaultAsyncHttpxClient,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from transformers import AutoTokenizer


@dataclass
class LLMResponseParts:
    """Structured view of the first assistant message in a chat completion."""

    content: str | None
    reasoning_content: str | None
    tool_calls: list[Any] | None = None
    parsed_json: Any | None = None
    message: Any | None = None
    response: Any | None = None


class LLMAbortError(Exception):
    """Base error that should not be swallowed by fallback logic."""


class APITerminatedError(LLMAbortError):
    """Raised when the user chooses to terminate after retryable API errors."""


class LLMRetryableError(LLMAbortError):
    """Raised when retryable API errors are exhausted without recovery."""


class LLMNonRetryableError(LLMAbortError):
    """Raised for configuration, request, parsing, or other non-retryable errors."""


class LLMSkippableError(Exception):
    """Raised for data-specific LLM errors that should skip the current item."""
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(override=True)

_LOG_DIR = os.getenv("OASIS_LOG_DIR", "outputs/logs")
os.makedirs(_LOG_DIR, exist_ok=True)
llm_log = logging.getLogger("oasis.llm")
llm_log.setLevel(logging.INFO)
if not any(isinstance(handler, logging.FileHandler) for handler in llm_log.handlers):
    _now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _llm_log_path = os.getenv(
        "OASIS_LLM_LOG_PATH",
        f"{_LOG_DIR}/oasis.llm-{_now}.log",
    )
    Path(_llm_log_path).parent.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.FileHandler(
        _llm_log_path,
        encoding="utf-8",
    )
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    llm_log.addHandler(_file_handler)

T = TypeVar("T")
_DEFAULT_USAGE_LOG_PATH = object()
_USAGE_LOG_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_DEFAULT_USAGE_LOG_FILE = Path(_LOG_DIR) / f"llm-usage-summary-{_USAGE_LOG_TIMESTAMP}.json"
_USAGE_SUMMARY_LOCK = threading.Lock()
_USAGE_SUMMARIES_BY_PATH: dict[Path, dict[tuple[str, str, str], dict[str, Any]]] = {}


class LLMModel:
    """统一封装 OpenAI-compatible LLM 调用、重试和 token 使用记录。"""

    retry_attempts = 5
    retry_delay = 2
    non_retryable_api_errors = (
        AuthenticationError,
        BadRequestError,
        NotFoundError,
        PermissionDeniedError,
        UnprocessableEntityError,
    )
    retryable_status_codes = {408, 429}
    skippable_api_error_codes = {
        "context_length_exceeded",
        "data_inspection_failed",
    }
    skippable_api_error_types = {"data_inspection_failed"}
    skippable_api_error_message_markers = (
        "context length",
        "inappropriate content",
        "input length",
        "maximum context length",
        "data_inspection_failed",
        "data inspection",
    )

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        max_memory_tokens: int = 5000,
        backend: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        enable_thinking: bool | None = None,
        usage_log_path: str | os.PathLike[str] | None | object = _DEFAULT_USAGE_LOG_PATH,
        timeout: float | None = 120,
        client_max_retries: int = 0,
        default_headers: dict[str, str] | None = None,
    ):
        """初始化 OpenAI-compatible 异步 LLM 客户端。

        参数:
            api_key: API 密钥。为空时按 backend 自动读取环境变量：
                DashScope 读取 DASHSCOPE_API_KEY，OpenAI 读取 OPENAI_API_KEY。
            base_url: API 基础地址。为空时按 backend 使用默认地址或环境变量。
            model_name: 实际请求的模型名。
            is_vllm: 是否使用本地 vLLM/transformers tokenizer 路径计数。
            max_memory_tokens: agent 记忆拼接时允许使用的最大 token 数。
            backend: 后端类型，可选 dashscope、openai。
            model_canonical: 用于统计和日志聚合的规范模型名。
            usage_log_path: token 使用汇总 JSON 记录路径；None 表示只保留内存记录。
            temperature: 默认采样温度。None 表示不向 provider 显式传递 temperature。
            top_p: 默认 nucleus sampling 参数。None 表示不向 provider 显式传递 top_p。
            enable_thinking: 是否开启混合思考模型的思考模式。False 表示关闭；
                None 表示不向 provider 显式传递该参数。
            timeout: OpenAI SDK 请求超时时间。
            client_max_retries: OpenAI SDK 内置重试次数；默认关闭，使用本类重试。
            default_headers: 创建 AsyncOpenAI 客户端时附加的默认请求头。
        """
        backend = backend or os.getenv("BACKEND", "dashscope")
        self.backend = backend
        if self.backend not in {"dashscope", "openai"}:
            raise ValueError("backend must be one of: dashscope, openai")

        self.model_name = model_name or os.getenv("MODEL_NAME", None)
        if not self.model_name:
            raise ValueError("model_name must be provided or set in MODEL_NAME")
        self.api_key = api_key or self._default_api_key()
        if not self.api_key:
            raise ValueError(
                "API key is required. Pass api_key or set DASHSCOPE_API_KEY "
                "or OPENAI_API_KEY for the selected backend."
            )

        self.base_url = base_url if base_url is not None else self._default_base_url()
        self.is_vllm =  os.getenv("IS_VLLM", "false").lower() == "true"
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.http_client = self._build_http_client(timeout)

        self.llm_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=client_max_retries,
            default_headers=default_headers,
            http_client=self.http_client,
        )
        self.max_memory_tokens = max_memory_tokens
        self.is_vllm = self.is_vllm
        self.response_history = []
        self.usage_history: list[dict[str, Any]] = []
        self.usage_log_path = self._resolve_usage_log_path(usage_log_path)
        if self.usage_log_path is not None:
            self.usage_log_path.parent.mkdir(parents=True, exist_ok=True)

        if self.is_vllm:
            self.tokenizer = self._load_vllm_tokenizer()
        else:
            self.tokenizer = self._get_encoding()

    def _load_vllm_tokenizer(self):
        tokenizer_root = Path(os.getenv("OASIS_MODEL_DIR", "models"))
        tokenizer_path = Path(
            os.getenv(
                "VLLM_TOKENIZER_PATH",
                str(tokenizer_root / self.model_name),
            )
        )
        if not tokenizer_path.is_dir():
            llm_log.warning(
                "Tokenizer path %s not found; falling back to tiktoken.",
                tokenizer_path,
            )
            return self._get_encoding()
        try:
            return AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
            )
        except Exception:
            llm_log.exception(
                "Failed to load tokenizer from %s; falling back to tiktoken.",
                tokenizer_path,
            )
            return self._get_encoding()

    def _default_api_key(self) -> str | None:
        """根据 backend 返回对应环境变量中的默认 API key。"""
        if self.backend == "openai":
            return os.getenv("OPENAI_API_KEY")
        return os.getenv("DASHSCOPE_API_KEY")

    def _default_base_url(self) -> str | None:
        """根据 backend 返回默认 base_url，允许通过环境变量覆盖。"""
        if self.backend == "openai":
            return os.getenv("OPENAI_BASE_URL")
        return os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def _resolve_usage_log_path(
        self,
        usage_log_path: str | os.PathLike[str] | None | object,
    ) -> Path | None:
        """解析 token 使用汇总输出路径。"""
        if usage_log_path is None:
            return None
        if usage_log_path is _DEFAULT_USAGE_LOG_PATH:
            env_path = os.getenv("OASIS_LLM_USAGE_LOG_PATH")
            if env_path:
                return Path(env_path)
            return _DEFAULT_USAGE_LOG_FILE
        return Path(usage_log_path)

    def _get_encoding(self):
        """获取当前模型对应的 tiktoken 编码器，默认 cl100k_base。"""
        return tiktoken.get_encoding("cl100k_base")

    def _build_http_client(self, timeout: float | None):
        """为本地 OpenAI-compatible 服务绕开环境代理。

        httpx/OpenAI SDK 默认读取 HTTP_PROXY/HTTPS_PROXY。当前机器上这些
        代理会截获 127.0.0.1 的 vLLM 请求并返回 502；curl 对 localhost
        通常直连，所以会表现为 curl 成功、SDK 失败。
        """
        if not self._uses_loopback_base_url():
            return None
        return DefaultAsyncHttpxClient(timeout=timeout, trust_env=False)

    def _uses_loopback_base_url(self) -> bool:
        if not self.base_url:
            return False
        parsed = urlparse(str(self.base_url))
        host = (parsed.hostname or "").lower()
        return host in {"127.0.0.1", "localhost", "::1"}

    def set_thinking(self, enabled: bool | None) -> None:
        """动态设置混合思考模型的思考开关；None 表示不显式传递。"""
        self.enable_thinking = enabled

    def disable_thinking(self) -> None:
        """关闭支持混合思考模式模型的思考输出。"""
        self.set_thinking(False)

    def enable_thinking_mode(self) -> None:
        """开启支持混合思考模式模型的思考输出。"""
        self.set_thinking(True)

    def get_tokens_num(self, text: str) -> int:
        """计算文本 token 数，用于记忆预算和上下文裁剪。"""
        return len(self.tokenizer.encode(text or ""))

    def _is_retryable_api_error(self, error: Exception) -> bool:
        """判断 API 异常是否属于重试可能修复的临时错误。"""
        if isinstance(error, self.non_retryable_api_errors):
            return False

        if isinstance(
            error,
            (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                InternalServerError,
            ),
        ):
            return True

        if isinstance(error, APIStatusError):
            return error.status_code in self.retryable_status_codes or (
                error.status_code >= 500
            )

        return False

    def _api_error_body(self, error: Exception) -> dict[str, Any]:
        """从 OpenAI SDK 异常中提取 provider 返回的原始错误 body。"""
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            return body

        response = getattr(error, "response", None)
        if response is not None:
            try:
                response_body = response.json()
                if isinstance(response_body, dict):
                    return response_body
            except Exception:
                return {}

        return {}

    def _extract_api_error_info(self, error: Exception) -> dict[str, Any]:
        """提取错误码、类型、消息、request_id 等结构化信息用于日志。"""
        body = self._api_error_body(error)
        provider_error = body.get("error", {}) if isinstance(body, dict) else {}
        if not isinstance(provider_error, dict):
            provider_error = {}

        return {
            "status_code": getattr(error, "status_code", None),
            "code": provider_error.get("code") or getattr(error, "code", None),
            "type": provider_error.get("type") or getattr(error, "type", None),
            "message": provider_error.get("message") or str(error),
            "param": provider_error.get("param"),
            "request_id": body.get("request_id") if isinstance(body, dict) else None,
            "response_id": body.get("id") if isinstance(body, dict) else None,
        }

    def _is_skippable_api_error(self, error: Exception) -> bool:
        """判断 API 异常是否属于输入数据导致、可跳过当前项的错误。"""
        if not isinstance(error, BadRequestError):
            return False

        info = self._extract_api_error_info(error)
        code = str(info.get("code") or "").lower()
        error_type = str(info.get("type") or "").lower()
        message = str(info.get("message") or "").lower()

        return (
            code in self.skippable_api_error_codes
            or error_type in self.skippable_api_error_types
            or any(
                marker in message
                for marker in self.skippable_api_error_message_markers
            )
        )

    def _to_skippable_error(
        self, operation_name: str, error: Exception
    ) -> LLMSkippableError:
        """记录可跳过错误日志，并包装为 LLMSkippableError。"""
        info = self._extract_api_error_info(error)
        message = (
            f"{operation_name} skipped because provider rejected the input. "
            f"status_code={info.get('status_code')}, code={info.get('code')}, "
            f"type={info.get('type')}, request_id={info.get('request_id')}, "
            f"response_id={info.get('response_id')}, message={info.get('message')}"
        )
        llm_log.warning(
            "[LLM_SKIPPABLE_ERROR] 跳过当前 LLM 调用；operation=%s, backend=%s, "
            "model=%s, status_code=%s, code=%s, type=%s, request_id=%s, "
            "response_id=%s, message=%s",
            operation_name,
            self.backend,
            self.model_name,
            info.get("status_code"),
            info.get("code"),
            info.get("type"),
            info.get("request_id"),
            info.get("response_id"),
            info.get("message"),
        )
        return LLMSkippableError(message)

    def _to_non_retryable_error(
        self, operation_name: str, error: Exception
    ) -> LLMNonRetryableError:
        """将不可重试异常包装为统一的 LLMNonRetryableError。"""
        return LLMNonRetryableError(
            f"{operation_name} failed with non-retryable error "
            f"{type(error).__name__}: {error}"
        )

    async def _call_with_retry(
        self,
        operation_name: str,
        request_factory: Callable[[], Awaitable[T]],
    ) -> T:
        """执行异步 API 请求，并只对可恢复错误进行重试。"""
        attempt = 0
        while True:
            try:
                return await request_factory()
            except LLMAbortError:
                raise
            except Exception as e:
                if self._is_skippable_api_error(e):
                    raise self._to_skippable_error(operation_name, e) from e

                if not self._is_retryable_api_error(e):
                    raise self._to_non_retryable_error(operation_name, e) from e

                attempt += 1
                if attempt < self.retry_attempts:
                    print(
                        f"第 {attempt} 次重试失败，"
                        f"{self.retry_delay} 秒后进行第 {attempt + 1} 次重试..."
                    )
                    await asyncio.sleep(self.retry_delay)
                    continue

                raise LLMRetryableError(
                    f"{operation_name} failed after {attempt} retryable "
                    f"API attempts with {type(e).__name__}: {e}"
                ) from e

    async def _chat_completion(
        self,
        operation_name: str,
        messages: list[dict[str, Any]],
        enable_thinking: bool | None = None,
        **kwargs,
    ):
        """统一发起 chat completion 请求并记录 response usage。"""
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            request_kwargs["top_p"] = self.top_p
        request_kwargs.update(kwargs)
        self._apply_thinking_options(request_kwargs, enable_thinking)

        async def request():
            """执行实际的 OpenAI SDK chat completion 请求。"""
            return await self.llm_client.chat.completions.create(**request_kwargs)

        response = await self._call_with_retry(operation_name, request)
        self._record_response_usage(operation_name, response)
        return response

    def _apply_thinking_options(
        self,
        request_kwargs: dict[str, Any],
        enable_thinking: bool | None = None,
    ) -> None:
        """按后端和模型格式写入思考模式开关。"""
        effective_enable_thinking = (
            self.enable_thinking if enable_thinking is None else enable_thinking
        )
        if effective_enable_thinking is None:
            return

        extra_body = dict(request_kwargs.get("extra_body") or {})
        if self.is_vllm:
            model_name = self.model_name.lower()
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            if "gpt" in model_name:
                extra_body["reasoning_effort"] = "low"
                chat_template_kwargs["reasoning_effort"] = "low"
            else:
                chat_template_kwargs["enable_thinking"] = effective_enable_thinking
            extra_body["chat_template_kwargs"] = chat_template_kwargs
        else:
            extra_body["enable_thinking"] = effective_enable_thinking
        request_kwargs["extra_body"] = extra_body

    def _jsonable(self, value: Any) -> Any:
        """将 OpenAI SDK/Pydantic 对象递归转换为可 JSON 序列化的数据。"""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, dict):
            return {k: self._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(v) for v in value]
        if hasattr(value, "__dict__"):
            return {
                k: self._jsonable(v)
                for k, v in vars(value).items()
                if not k.startswith("_")
            }
        return str(value)

    def _coerce_int(self, value: Any) -> int:
        """将可能为空或非数字的值安全转换为 int。"""
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _coerce_optional_int(self, value: Any) -> int | None:
        """将可选数字字段安全转换为 int，空值保持为 None。"""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_float(self, value: Any) -> float:
        """将可能为空或非数字的值安全转换为 float。"""
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _record_response_usage(self, operation_name: str, response: Any) -> None:
        """从 response.usage 中提取 token 信息并写入内存和汇总文件。"""
        self.response_history.append(response)

        usage = getattr(response, "usage", None)
        usage_dict = self._jsonable(usage) or {}
        if not isinstance(usage_dict, dict):
            usage_dict = {}
        prompt_tokens = self._coerce_int(usage_dict.get("prompt_tokens"))
        completion_tokens = self._coerce_int(usage_dict.get("completion_tokens"))
        total_tokens = self._coerce_int(usage_dict.get("total_tokens"))
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        record = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "operation": operation_name,
            "backend": self.backend,
            "model": self.model_name,
            "response_id": getattr(response, "id", None),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": self._coerce_float(usage_dict.get("cost")),
            "usage": usage_dict,
        }

        self.usage_history.append(record)
        if self.usage_log_path is not None:
            self._update_usage_summary_file(
                operation_name,
                prompt_tokens,
                completion_tokens,
            )

    def _load_usage_summaries_from_file(
        self,
        path: Path,
    ) -> dict[tuple[str, str, str], dict[str, Any]]:
        """读取已有 token 汇总文件，用于中断续跑后继续累加。"""
        if not path.exists():
            return {}

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        groups = payload.get("groups") if isinstance(payload, dict) else None
        if not isinstance(groups, list):
            return {}

        summaries: dict[tuple[str, str, str], dict[str, Any]] = {}
        for group in groups:
            if not isinstance(group, dict):
                continue

            operation = str(group.get("operation") or "unknown")
            backend = str(group.get("backend") or "unknown")
            model = str(group.get("model") or "unknown")
            requests = self._coerce_int(group.get("requests"))
            input_total = self._coerce_int(group.get("input_tokens_total"))
            output_total = self._coerce_int(group.get("output_tokens_total"))
            input_min = self._coerce_optional_int(group.get("input_tokens_min"))
            input_max = self._coerce_optional_int(group.get("input_tokens_max"))
            output_min = self._coerce_optional_int(group.get("output_tokens_min"))
            output_max = self._coerce_optional_int(group.get("output_tokens_max"))

            if requests <= 0:
                continue

            summaries[(operation, backend, model)] = {
                "operation": operation,
                "backend": backend,
                "model": model,
                "requests": requests,
                "input_tokens_total": input_total,
                "output_tokens_total": output_total,
                "input_tokens_min": input_min if input_min is not None else input_total,
                "input_tokens_max": input_max if input_max is not None else input_total,
                "output_tokens_min": (
                    output_min if output_min is not None else output_total
                ),
                "output_tokens_max": (
                    output_max if output_max is not None else output_total
                ),
            }
        return summaries

    def _usage_summaries_for_path(
        self,
        path: Path,
    ) -> dict[tuple[str, str, str], dict[str, Any]]:
        if path not in _USAGE_SUMMARIES_BY_PATH:
            _USAGE_SUMMARIES_BY_PATH[path] = self._load_usage_summaries_from_file(path)
        return _USAGE_SUMMARIES_BY_PATH[path]

    def _update_usage_summary_file(
        self,
        operation_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """按 operation/backend/model 聚合 token 用量并覆盖写入汇总文件。"""
        if self.usage_log_path is None:
            return

        path = self.usage_log_path
        key = (operation_name, self.backend, self.model_name)
        with _USAGE_SUMMARY_LOCK:
            summaries = self._usage_summaries_for_path(path)
            summary = summaries.setdefault(
                key,
                {
                    "operation": operation_name,
                    "backend": self.backend,
                    "model": self.model_name,
                    "requests": 0,
                    "input_tokens_total": 0,
                    "output_tokens_total": 0,
                    "input_tokens_min": None,
                    "input_tokens_max": None,
                    "output_tokens_min": None,
                    "output_tokens_max": None,
                },
            )
            summary["requests"] += 1
            summary["input_tokens_total"] += prompt_tokens
            summary["output_tokens_total"] += completion_tokens
            summary["input_tokens_min"] = (
                prompt_tokens
                if summary["input_tokens_min"] is None
                else min(summary["input_tokens_min"], prompt_tokens)
            )
            summary["input_tokens_max"] = (
                prompt_tokens
                if summary["input_tokens_max"] is None
                else max(summary["input_tokens_max"], prompt_tokens)
            )
            summary["output_tokens_min"] = (
                completion_tokens
                if summary["output_tokens_min"] is None
                else min(summary["output_tokens_min"], completion_tokens)
            )
            summary["output_tokens_max"] = (
                completion_tokens
                if summary["output_tokens_max"] is None
                else max(summary["output_tokens_max"], completion_tokens)
            )

            payload = {
                "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "groups": [
                    self._format_usage_summary(group)
                    for group in sorted(
                        summaries.values(),
                        key=lambda item: (
                            item["operation"],
                            item["backend"],
                            item["model"],
                        ),
                    )
                ],
            }
            tmp_path = path.with_name(f".{path.name}.tmp")
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            tmp_path.replace(path)

    def _format_usage_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        requests = summary["requests"]
        return {
            "operation": summary["operation"],
            "backend": summary["backend"],
            "model": summary["model"],
            "requests": requests,
            "input_tokens_total": summary["input_tokens_total"],
            "output_tokens_total": summary["output_tokens_total"],
            "input_tokens_avg": summary["input_tokens_total"] / requests,
            "output_tokens_avg": summary["output_tokens_total"] / requests,
            "input_tokens_min": summary["input_tokens_min"],
            "input_tokens_max": summary["input_tokens_max"],
            "output_tokens_min": summary["output_tokens_min"],
            "output_tokens_max": summary["output_tokens_max"],
        }

    def get_token_usage_summary(self) -> dict[str, Any]:
        """汇总当前进程内累计的请求次数、token 数和成本。"""
        summary = {
            "requests": len(self.usage_history),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "by_model": {},
        }

        for record in self.usage_history:
            summary["prompt_tokens"] += record["prompt_tokens"]
            summary["completion_tokens"] += record["completion_tokens"]
            summary["total_tokens"] += record["total_tokens"]
            summary["total_cost"] += record["cost"]

            model_key = record["model"]
            model_summary = summary["by_model"].setdefault(
                model_key,
                {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                },
            )
            model_summary["requests"] += 1
            model_summary["prompt_tokens"] += record["prompt_tokens"]
            model_summary["completion_tokens"] += record["completion_tokens"]
            model_summary["total_tokens"] += record["total_tokens"]
            model_summary["total_cost"] += record["cost"]

        return summary

    def get_running_cost_num_prompt_completion_tokens(self):
        """返回历史兼容格式的累计成本、prompt token 和 completion token。"""
        summary = self.get_token_usage_summary()
        return (
            summary["total_cost"],
            summary["prompt_tokens"],
            summary["completion_tokens"],
        )

    def _first_message(self, operation_name: str, response: Any) -> Any:
        """读取 response 中第一条 assistant message。"""
        try:
            return response.choices[0].message
        except Exception as e:
            raise LLMNonRetryableError(
                f"{operation_name} failed to read response message: "
                f"{response}. Error: {e}"
            ) from e

    def _read_message_content(self, operation_name: str, response: Any) -> str:
        """读取 response 中第一条 assistant message 的 content。"""
        content = self._read_message_parts(
            operation_name,
            response,
            strip_inline_reasoning=False,
        ).content

        if content is None:
            raise LLMNonRetryableError(
                f"{operation_name} failed because response content is None: "
                f"{response}"
            )
        return content

    def _message_dicts(self, message: Any) -> list[dict[str, Any]]:
        """从 SDK/Pydantic message 对象中收集可查找的字段字典。"""
        dicts: list[dict[str, Any]] = []
        if isinstance(message, dict):
            dicts.append(message)

        model_dump = getattr(message, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump(mode="json", exclude_none=True)
            except TypeError:
                dumped = model_dump()
            if isinstance(dumped, dict):
                dicts.append(dumped)

        model_extra = getattr(message, "model_extra", None)
        if isinstance(model_extra, dict):
            dicts.append(model_extra)

        try:
            vars_dict = vars(message)
        except TypeError:
            vars_dict = {}
        if isinstance(vars_dict, dict):
            dicts.append(vars_dict)

        return dicts

    def _message_value(self, message: Any, key: str, default: Any = None) -> Any:
        """兼容属性、dict、model_extra 和 vars(message) 读取 message 字段。"""
        if isinstance(message, dict) and key in message:
            return message[key]

        value = getattr(message, key, default)
        if value is not default:
            return value

        for data in self._message_dicts(message):
            if key in data:
                return data[key]
        return default

    def _coerce_optional_text(self, value: Any) -> str | None:
        """将 provider 的 reasoning/content 字段转成可记录文本。"""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list, tuple)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except TypeError:
                return str(value)
        return str(value)

    def _split_inline_reasoning(
        self,
        content: str | None,
    ) -> tuple[str | None, str | None]:
        """从 content 中拆出常见的内联 reasoning 标签。"""
        if content is None:
            return None, None

        patterns = (
            r"<think>\s*(.*?)\s*</think>",
            r"<reasoning>\s*(.*?)\s*</reasoning>",
        )
        reasoning_chunks: list[str] = []
        cleaned_content = content
        for pattern in patterns:
            matches = re.findall(pattern, cleaned_content, flags=re.DOTALL | re.I)
            reasoning_chunks.extend(chunk.strip() for chunk in matches if chunk.strip())
            cleaned_content = re.sub(
                pattern,
                "",
                cleaned_content,
                flags=re.DOTALL | re.I,
            )

        reasoning = "\n\n".join(reasoning_chunks) if reasoning_chunks else None
        return cleaned_content.strip(), reasoning

    def _read_message_parts(
        self,
        operation_name: str,
        response: Any,
        *,
        strip_inline_reasoning: bool = True,
    ) -> LLMResponseParts:
        """读取最终内容、reasoning、tool calls 和原始对象。"""
        message = self._first_message(operation_name, response)
        content = self._coerce_optional_text(
            self._message_value(message, "content", None)
        )
        reasoning_content = self._coerce_optional_text(
            self._message_value(message, "reasoning_content", None)
        )
        if reasoning_content is None:
            reasoning_content = self._coerce_optional_text(
                self._message_value(message, "reasoning", None)
            )

        if reasoning_content is None and strip_inline_reasoning:
            content, reasoning_content = self._split_inline_reasoning(content)

        tool_calls = self._message_value(message, "tool_calls", None)

        return LLMResponseParts(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            message=message,
            response=response,
        )

    async def get_tool_call_response(self, messages, tools):
        """请求模型生成工具调用响应，返回完整 chat completion response。"""
        return await self._chat_completion(
            "get_tool_call_response",
            messages,
            tools=tools,
            parallel_tool_calls=False,
        )

    async def get_tool_call_response_parts(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        enable_thinking: bool | None = None,
    ) -> LLMResponseParts:
        """请求模型生成工具调用响应，并返回拆分后的内容和 reasoning。"""
        response = await self._chat_completion(
            "get_tool_call_response_parts",
            messages,
            tools=tools,
            parallel_tool_calls=False,
            enable_thinking=enable_thinking,
        )
        return self._read_message_parts("get_tool_call_response_parts", response)

    async def get_message_response_parts(
        self,
        messages: list[dict[str, Any]],
        *,
        enable_thinking: bool | None = None,
        operation_name: str = "get_message_response_parts",
    ) -> LLMResponseParts:
        """请求普通 chat 响应，并返回拆分后的内容和 reasoning。"""
        response = await self._chat_completion(
            operation_name,
            messages,
            stream=False,
            enable_thinking=enable_thinking,
        )
        return self._read_message_parts(operation_name, response)

    async def get_json_response_parts(
        self,
        user_prompt: str,
        *,
        enable_thinking: bool | None = None,
        operation_name: str = "get_json_response_parts",
    ) -> LLMResponseParts:
        """请求模型返回 JSON 对象，并保留 reasoning 与解析结果。"""
        last_response = None
        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            response = await self._chat_completion(
                operation_name,
                [
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                stream=False,
                enable_thinking=enable_thinking,
            )
            last_response = response
            parts = self._read_message_parts(operation_name, response)
            try:
                parts.parsed_json = json.loads(parts.content or "{}")
                return parts
            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts:
                    llm_log.warning(
                        "%s failed to parse JSON response on attempt %s/%s: %s",
                        operation_name,
                        attempt,
                        self.retry_attempts,
                        e,
                    )
                    await asyncio.sleep(self.retry_delay)
                    continue

        raise LLMNonRetryableError(
            f"{operation_name} failed to parse JSON response after "
            f"{self.retry_attempts} attempts: {last_response}. Error: {last_error}"
        ) from last_error

    async def get_json_response(self, user_prompt):
        """请求模型返回 JSON 对象，并解析为 Python dict/list 等结构。"""
        parts = await self.get_json_response_parts(
            user_prompt,
            operation_name="get_json_response",
        )
        return parts.parsed_json

    async def get_text_response_parts(
        self,
        text: str,
        *,
        enable_thinking: bool | None = None,
        operation_name: str = "get_text_response_parts",
    ) -> LLMResponseParts:
        """请求模型返回文本，并保留 reasoning。"""
        response = await self._chat_completion(
            operation_name,
            [
                {"role": "user", "content": text},
            ],
            stream=False,
            presence_penalty=1.5,
            enable_thinking=enable_thinking,
        )
        return self._read_message_parts(operation_name, response)

    async def get_pure_text_response(
        self,
        text: str,
        *,
        enable_thinking: bool | None = None,
    ):
        """请求模型返回纯文本内容，用于无需工具调用的场景。"""
        parts = await self.get_text_response_parts(
            text,
            enable_thinking=enable_thinking,
            operation_name="get_pure_text_response",
        )
        if parts.content is None:
            raise LLMNonRetryableError(
                f"get_pure_text_response failed because response content is None: "
                f"{parts.response}"
            )
        return parts.content
