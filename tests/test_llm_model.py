import asyncio
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import APIConnectionError, BadRequestError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OASIS_ROOT = PROJECT_ROOT / "oasis"

oasis_pkg = types.ModuleType("oasis")
oasis_pkg.__path__ = [str(OASIS_ROOT)]
sys.modules.setdefault("oasis", oasis_pkg)

social_agent_pkg = types.ModuleType("oasis.social_agent")
social_agent_pkg.__path__ = [str(OASIS_ROOT / "social_agent")]
sys.modules.setdefault("oasis.social_agent", social_agent_pkg)

transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoTokenizer = SimpleNamespace(
    from_pretrained=lambda *args, **kwargs: None,
)
sys.modules.setdefault("transformers", transformers_stub)


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


llm_model = load_module(
    "oasis.social_agent.llm_model",
    OASIS_ROOT / "social_agent" / "llm_model.py",
)

LLMNonRetryableError = llm_model.LLMNonRetryableError
LLMSkippableError = llm_model.LLMSkippableError
LLMResponseParts = llm_model.LLMResponseParts
LLMModel = llm_model.LLMModel


class FakeCompletions:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    def __init__(self, results):
        self.completions = FakeCompletions(results)
        self.chat = SimpleNamespace(completions=self.completions)


def make_response(
    content="ok",
    usage=None,
    response_id="resp_1",
    message=None,
    reasoning_content=None,
    reasoning=None,
    tool_calls=None,
):
    if message is None:
        message_kwargs = {"content": content}
        if reasoning_content is not None:
            message_kwargs["reasoning_content"] = reasoning_content
        if reasoning is not None:
            message_kwargs["reasoning"] = reasoning
        if tool_calls is not None:
            message_kwargs["tool_calls"] = tool_calls
        message = SimpleNamespace(**message_kwargs)
    return SimpleNamespace(
        id=response_id,
        choices=[
            SimpleNamespace(
                message=message,
            )
        ],
        usage=usage or {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
            "cost": 0.25,
        },
    )


def make_connection_error():
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    return APIConnectionError(message="temporary network error", request=request)


def make_bad_request_error():
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError("bad request", response=response, body=None)


def make_data_inspection_error():
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    body = {
        "error": {
            "message": "Input data may contain inappropriate content.",
            "type": "data_inspection_failed",
            "param": None,
            "code": "data_inspection_failed",
        },
        "id": "chatcmpl-test",
        "request_id": "request-test",
    }
    response = httpx.Response(400, request=request, json=body)
    return BadRequestError("data inspection failed", response=response, body=body)


def make_context_length_error():
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    body = {
        "error": {
            "message": (
                "Input length (8425) exceeds model's maximum context length "
                "(8192)."
            ),
            "type": "BadRequestError",
            "param": None,
            "code": 400,
        }
    }
    response = httpx.Response(400, request=request, json=body)
    return BadRequestError("context length exceeded", response=response, body=body)


class LLMModelTest(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(
            "os.environ",
            {
                "BACKEND": "dashscope",
                "MODEL_NAME": "qwen-plus-latest",
                "IS_VLLM": "false",
            },
        )
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_legacy_dashscope_construction_still_works(self):
        llm = LLMModel(api_key="test-key", usage_log_path=None)

        self.assertEqual(llm.backend, "dashscope")
        self.assertEqual(llm.model_name, "qwen-plus-latest")
        self.assertEqual(
            llm.base_url,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def test_sampling_params_are_passed_to_chat_completion(self):
        llm = LLMModel(
            api_key="test-key",
            temperature=0.4,
            top_p=0.9,
            usage_log_path=None,
        )
        fake_client = FakeClient([make_response()])
        llm.llm_client = fake_client

        asyncio.run(llm.get_pure_text_response("hello"))

        call = fake_client.completions.calls[0]
        self.assertEqual(call["temperature"], 0.4)
        self.assertEqual(call["top_p"], 0.9)

    def test_disable_thinking_is_passed_to_dashscope_extra_body(self):
        with patch.dict("os.environ", {"IS_VLLM": "false"}):
            llm = LLMModel(
                api_key="test-key",
                enable_thinking=False,
                usage_log_path=None,
            )
        fake_client = FakeClient([make_response()])
        llm.llm_client = fake_client

        asyncio.run(llm.get_pure_text_response("hello"))

        call = fake_client.completions.calls[0]
        self.assertEqual(call["extra_body"], {"enable_thinking": False})

    def test_disable_thinking_is_passed_to_vllm_chat_template_kwargs(self):
        llm = LLMModel(
            api_key="test-key",
            enable_thinking=False,
            usage_log_path=None,
        )
        llm.is_vllm = True
        fake_client = FakeClient([make_response()])
        llm.llm_client = fake_client

        asyncio.run(llm.get_pure_text_response("hello"))

        call = fake_client.completions.calls[0]
        self.assertEqual(
            call["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_thinking_options_preserve_existing_extra_body(self):
        with patch.dict("os.environ", {"IS_VLLM": "false"}):
            llm = LLMModel(
                api_key="test-key",
                enable_thinking=False,
                usage_log_path=None,
            )
        fake_client = FakeClient([make_response()])
        llm.llm_client = fake_client

        asyncio.run(
            llm._chat_completion(
                "test_operation",
                [{"role": "user", "content": "hello"}],
                extra_body={"metadata": {"source": "unit-test"}},
            )
        )

        call = fake_client.completions.calls[0]
        self.assertEqual(
            call["extra_body"],
            {
                "metadata": {"source": "unit-test"},
                "enable_thinking": False,
            },
        )

    def test_openrouter_backend_is_not_supported(self):
        with patch.dict("os.environ", {"BACKEND": ""}):
            with self.assertRaisesRegex(ValueError, "dashscope, openai"):
                LLMModel(
                    api_key="test-key",
                    backend="openrouter",
                    usage_log_path=None,
                )

    def test_success_records_usage_in_memory_and_summary_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            usage_log_path = Path(temp_dir) / "usage.json"
            llm = LLMModel(
                api_key="test-key",
                usage_log_path=usage_log_path,
            )
            llm.llm_client = FakeClient([make_response()])

            result = asyncio.run(llm.get_pure_text_response("hello"))
            summary = llm.get_token_usage_summary()
            cost, prompt_tokens, completion_tokens = (
                llm.get_running_cost_num_prompt_completion_tokens()
            )

            self.assertEqual(result, "ok")
            self.assertEqual(len(llm.response_history), 1)
            self.assertEqual(len(llm.usage_history), 1)
            self.assertEqual(summary["requests"], 1)
            self.assertEqual(summary["prompt_tokens"], 3)
            self.assertEqual(summary["completion_tokens"], 2)
            self.assertEqual(summary["total_tokens"], 5)
            self.assertEqual(summary["total_cost"], 0.25)
            self.assertEqual((cost, prompt_tokens, completion_tokens), (0.25, 3, 2))

            usage_summary = json.loads(usage_log_path.read_text(encoding="utf-8"))
            self.assertEqual(len(usage_summary["groups"]), 1)
            record = usage_summary["groups"][0]
            self.assertEqual(record["operation"], "get_pure_text_response")
            self.assertEqual(record["backend"], "dashscope")
            self.assertEqual(record["model"], "qwen-plus-latest")
            self.assertEqual(record["requests"], 1)
            self.assertEqual(record["input_tokens_total"], 3)
            self.assertEqual(record["output_tokens_total"], 2)
            self.assertEqual(record["input_tokens_avg"], 3)
            self.assertEqual(record["output_tokens_avg"], 2)
            self.assertEqual(record["input_tokens_min"], 3)
            self.assertEqual(record["input_tokens_max"], 3)
            self.assertEqual(record["output_tokens_min"], 2)
            self.assertEqual(record["output_tokens_max"], 2)

    def test_usage_summary_file_is_shared_by_log_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            usage_log_path = Path(temp_dir) / "usage.json"
            first = LLMModel(api_key="test-key", usage_log_path=usage_log_path)
            second = LLMModel(api_key="test-key", usage_log_path=usage_log_path)
            first.llm_client = FakeClient(
                [
                    make_response(
                        usage={
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                            "cost": 0.25,
                        }
                    )
                ]
            )
            second.llm_client = FakeClient(
                [
                    make_response(
                        content='{"ok": true}',
                        usage={
                            "prompt_tokens": 7,
                            "completion_tokens": 5,
                            "total_tokens": 12,
                            "cost": 0.5,
                        },
                    )
                ]
            )

            asyncio.run(first.get_pure_text_response("hello"))
            asyncio.run(second.get_json_response("return json"))

            usage_summary = json.loads(usage_log_path.read_text(encoding="utf-8"))
            groups = {
                group["operation"]: group
                for group in usage_summary["groups"]
            }

            self.assertEqual(set(groups), {"get_pure_text_response", "get_json_response"})
            self.assertEqual(groups["get_pure_text_response"]["input_tokens_total"], 3)
            self.assertEqual(groups["get_pure_text_response"]["output_tokens_total"], 2)
            self.assertEqual(groups["get_json_response"]["input_tokens_total"], 7)
            self.assertEqual(groups["get_json_response"]["output_tokens_total"], 5)

    def test_usage_summary_file_survives_process_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            usage_log_path = Path(temp_dir) / "usage.json"
            first = LLMModel(api_key="test-key", usage_log_path=usage_log_path)
            first.llm_client = FakeClient(
                [
                    make_response(
                        usage={
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                            "cost": 0.25,
                        }
                    )
                ]
            )
            asyncio.run(first.get_pure_text_response("hello"))

            llm_model._USAGE_SUMMARIES_BY_PATH.clear()

            second = LLMModel(api_key="test-key", usage_log_path=usage_log_path)
            second.llm_client = FakeClient(
                [
                    make_response(
                        usage={
                            "prompt_tokens": 7,
                            "completion_tokens": 5,
                            "total_tokens": 12,
                            "cost": 0.5,
                        }
                    )
                ]
            )
            asyncio.run(second.get_pure_text_response("hello again"))

            usage_summary = json.loads(usage_log_path.read_text(encoding="utf-8"))
            record = usage_summary["groups"][0]
            self.assertEqual(record["requests"], 2)
            self.assertEqual(record["input_tokens_total"], 10)
            self.assertEqual(record["output_tokens_total"], 7)
            self.assertEqual(record["input_tokens_min"], 3)
            self.assertEqual(record["input_tokens_max"], 7)
            self.assertEqual(record["output_tokens_min"], 2)
            self.assertEqual(record["output_tokens_max"], 5)

    def test_retryable_error_retries_then_succeeds(self):
        llm = LLMModel(
            api_key="test-key",
            usage_log_path=None,
        )
        llm.retry_attempts = 2
        llm.retry_delay = 0
        fake_client = FakeClient([make_connection_error(), make_response()])
        llm.llm_client = fake_client

        with contextlib.redirect_stdout(io.StringIO()):
            result = asyncio.run(llm.get_pure_text_response("hello"))

        self.assertEqual(result, "ok")
        self.assertEqual(len(fake_client.completions.calls), 2)

    def test_non_retryable_error_does_not_retry(self):
        llm = LLMModel(
            api_key="test-key",
            usage_log_path=None,
        )
        llm.retry_delay = 0
        fake_client = FakeClient([make_bad_request_error(), make_response()])
        llm.llm_client = fake_client

        with self.assertRaises(LLMNonRetryableError):
            asyncio.run(llm.get_pure_text_response("hello"))

        self.assertEqual(len(fake_client.completions.calls), 1)

    def test_data_inspection_error_is_skippable_and_logged(self):
        llm = LLMModel(
            api_key="test-key",
            usage_log_path=None,
        )
        fake_client = FakeClient([make_data_inspection_error(), make_response()])
        llm.llm_client = fake_client

        with self.assertLogs("oasis.llm", level="WARNING") as logs:
            with self.assertRaises(LLMSkippableError) as ctx:
                asyncio.run(llm.get_pure_text_response("bad content"))

        log_text = "\n".join(logs.output)
        self.assertIn("[LLM_SKIPPABLE_ERROR]", log_text)
        self.assertIn("data_inspection_failed", log_text)
        self.assertIn("request-test", log_text)
        self.assertIn("get_pure_text_response skipped", str(ctx.exception))
        self.assertEqual(len(fake_client.completions.calls), 1)
        self.assertEqual(len(llm.usage_history), 0)

    def test_context_length_error_is_skippable_for_tool_calls(self):
        llm = LLMModel(
            api_key="test-key",
            usage_log_path=None,
        )
        fake_client = FakeClient([make_context_length_error(), make_response()])
        llm.llm_client = fake_client

        with self.assertLogs("oasis.llm", level="WARNING") as logs:
            with self.assertRaises(LLMSkippableError) as ctx:
                asyncio.run(
                    llm.get_tool_call_response_parts(
                        [{"role": "user", "content": "act"}],
                        [{"type": "function", "function": {"name": "like"}}],
                    )
                )

        log_text = "\n".join(logs.output)
        self.assertIn("[LLM_SKIPPABLE_ERROR]", log_text)
        self.assertIn("maximum context length", log_text)
        self.assertIn("get_tool_call_response_parts skipped", str(ctx.exception))
        self.assertEqual(len(fake_client.completions.calls), 1)
        self.assertEqual(len(llm.usage_history), 0)

    def test_json_parse_error_is_non_retryable(self):
        llm = LLMModel(api_key="test-key", usage_log_path=None)
        llm.retry_attempts = 1
        fake_client = FakeClient([make_response(content="not json")])
        llm.llm_client = fake_client

        with self.assertRaises(LLMNonRetryableError):
            asyncio.run(llm.get_json_response("return json"))

        self.assertEqual(len(fake_client.completions.calls), 1)

    def test_text_response_parts_reads_reasoning_content(self):
        llm = LLMModel(api_key="test-key", usage_log_path=None)
        fake_client = FakeClient(
            [
                make_response(
                    content="final answer",
                    reasoning_content="hidden reasoning",
                )
            ]
        )
        llm.llm_client = fake_client

        parts = asyncio.run(
            llm.get_text_response_parts("hello", enable_thinking=True)
        )

        self.assertIsInstance(parts, LLMResponseParts)
        self.assertEqual(parts.content, "final answer")
        self.assertEqual(parts.reasoning_content, "hidden reasoning")
        self.assertIs(parts.response, llm.response_history[0])
        self.assertEqual(
            fake_client.completions.calls[0]["extra_body"],
            {"enable_thinking": True},
        )

    def test_response_parts_reads_reasoning_from_model_extra(self):
        llm = LLMModel(api_key="test-key", usage_log_path=None)
        message = SimpleNamespace(content="final answer")
        message.model_extra = {"reasoning": "extra reasoning"}
        fake_client = FakeClient([make_response(message=message)])
        llm.llm_client = fake_client

        parts = asyncio.run(llm.get_text_response_parts("hello"))

        self.assertEqual(parts.content, "final answer")
        self.assertEqual(parts.reasoning_content, "extra reasoning")

    def test_response_parts_strips_inline_reasoning(self):
        llm = LLMModel(api_key="test-key", usage_log_path=None)
        fake_client = FakeClient(
            [
                make_response(
                    content="<think>step one</think>\n{\"ok\": true}",
                )
            ]
        )
        llm.llm_client = fake_client

        parts = asyncio.run(llm.get_json_response_parts("return json"))

        self.assertEqual(parts.content, '{"ok": true}')
        self.assertEqual(parts.reasoning_content, "step one")
        self.assertEqual(parts.parsed_json, {"ok": True})

    def test_tool_call_response_parts_preserves_empty_content_and_tool_calls(self):
        llm = LLMModel(api_key="test-key", usage_log_path=None)
        tool_calls = [SimpleNamespace(id="call_1", function=SimpleNamespace(name="like"))]
        fake_client = FakeClient(
            [
                make_response(
                    content=None,
                    reasoning_content="choose a tool",
                    tool_calls=tool_calls,
                )
            ]
        )
        llm.llm_client = fake_client

        parts = asyncio.run(
            llm.get_tool_call_response_parts(
                [{"role": "user", "content": "act"}],
                [{"type": "function", "function": {"name": "like"}}],
            )
        )

        self.assertIsNone(parts.content)
        self.assertEqual(parts.reasoning_content, "choose a tool")
        self.assertEqual(parts.tool_calls, tool_calls)

    def test_tool_call_response_parts_does_not_parse_baseline_specific_content(self):
        llm = LLMModel(api_key="test-key", usage_log_path=None)
        content = (
            "<ANA>\n分析当前观察。\n</ANA>\n"
            "<MEM>\n比较历史记忆。\n</MEM>\n"
            'create_comment(post_id=101, content="请补充具体站点。")'
        )
        fake_client = FakeClient(
            [
                make_response(
                    content=content,
                    reasoning="provider reasoning",
                    tool_calls=[],
                )
            ]
        )
        llm.llm_client = fake_client

        parts = asyncio.run(
            llm.get_tool_call_response_parts(
                [{"role": "user", "content": "act"}],
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "create_comment",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )
        )

        self.assertEqual(parts.reasoning_content, "provider reasoning")
        self.assertEqual(parts.tool_calls, [])

    def test_loopback_base_url_disables_environment_proxy(self):
        with patch.dict(
            "os.environ",
            {
                "BACKEND": "openai",
                "MODEL_NAME": "local-model",
                "OPENAI_BASE_URL": "http://127.0.0.1:8009/v1",
                "IS_VLLM": "true",
            },
        ):
            llm = LLMModel(api_key="test-key", usage_log_path=None)

        self.assertIsNotNone(llm.http_client)


if __name__ == "__main__":
    unittest.main()
