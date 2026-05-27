# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from camel.agents import ChatAgent
from camel.memories import ChatHistoryMemory
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelManager
from camel.prompts import TextPrompt
from camel.toolkits import FunctionTool
from camel.types import OpenAIBackendRole

from oasis.localization import is_zh_locale, platform_display_name, zh_action_label
from oasis.social_agent.agent_common import agent_log
from oasis.social_agent.agent_action import SocialAction
from oasis.social_agent.agent_environment import SocialEnvironment
from oasis.social_agent.token_truncating_context import (
    RecentTokenLimitContextCreator,
)
from oasis.social_platform import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType

if TYPE_CHECKING:
    from oasis.social_agent.agent_graph import AgentGraph


class SocialAgent(ChatAgent):
    r"""Social Agent."""

    _TOOL_ACTION_NAMES = frozenset(SocialAction._action_function_names)

    @staticmethod
    def _tool_name(tool: dict) -> str:
        return tool["function"]["name"]

    @classmethod
    def _normalize_tool_action_name(cls, action_name: str) -> str:
        if not isinstance(action_name, str):
            return action_name
        action_name = action_name.strip()
        if action_name in cls._TOOL_ACTION_NAMES:
            return action_name

        # Some local tool parsers may leak channel/control tokens into the
        # function name, e.g. "follow<|channel|>commentary".
        if "<|" in action_name:
            candidate = action_name.split("<|", 1)[0].strip()
            if candidate in cls._TOOL_ACTION_NAMES:
                return candidate

        return action_name

    async def _aexecute_tool(self, tool_call_request):
        raw_tool_name = tool_call_request.tool_name
        tool_name = self._normalize_tool_action_name(raw_tool_name)
        if tool_name != raw_tool_name:
            agent_log.warning(
                f"Agent {self.social_agent_id} normalized tool action "
                f"name from {raw_tool_name!r} to {tool_name!r}."
            )
            if hasattr(tool_call_request, "model_copy"):
                tool_call_request = tool_call_request.model_copy(
                    update={"tool_name": tool_name}
                )
            else:
                tool_call_request = tool_call_request.copy(
                    update={"tool_name": tool_name}
                )
        return await super()._aexecute_tool(tool_call_request)

    @staticmethod
    def _summarize_value(value: Any, max_length: int = 80) -> Any:
        if isinstance(value, str):
            value = value.replace("\n", " ").strip()
            return value if len(value) <= max_length else f"{value[:max_length]}..."
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            summary = {}
            for key, val in list(value.items())[:5]:
                if "prompt" in str(key).lower():
                    summary[key] = f"<hidden, {len(str(val))} chars>"
                else:
                    summary[key] = SocialAgent._summarize_value(
                        val, max_length)
            return summary
        if isinstance(value, (list, tuple, set)):
            items = [
                SocialAgent._summarize_value(item, max_length)
                for item in list(value)[:5]
            ]
            if len(value) > 5:
                items.append(f"...({len(value)} total)")
            return items
        text = str(value).replace("\n", " ").strip()
        return text if len(text) <= max_length else f"{text[:max_length]}..."

    def _summarize_action_args(self,
                               args: Any = None,
                               kwargs: dict[str, Any] | None = None) -> str:
        summary: dict[str, Any] = {}
        if args:
            summary["args"] = self._summarize_value(args)
        if kwargs:
            summary["kwargs"] = self._summarize_value(kwargs)
        return str(summary or {})

    def _action_display_name(self, action_name: str) -> str:
        return zh_action_label(action_name) if is_zh_locale(
            self.locale) else action_name

    def _log_action_summary(self,
                            action_name: str,
                            args: Any = None,
                            kwargs: dict[str, Any] | None = None) -> None:
        agent_log.info(
            f"Agent {self.social_agent_id} action: "
            f"{self._action_display_name(action_name)}; "
            f"params: {self._summarize_action_args(args, kwargs)}")

    @staticmethod
    def _tool_call_args(tool_call: Any) -> Any:
        if isinstance(tool_call, dict):
            return tool_call.get("args")
        return getattr(tool_call, "args", None)

    @staticmethod
    def _tool_call_name(tool_call: Any) -> str:
        if isinstance(tool_call, dict):
            return str(tool_call.get("tool_name") or tool_call.get("name") or "")
        return str(
            getattr(tool_call, "tool_name", None)
            or getattr(tool_call, "name", None)
            or "")

    def __init__(self,
                 agent_id: int,
                 user_info: UserInfo,
                 user_info_template: TextPrompt | None = None,
                 channel: Channel | None = None,
                 model: Optional[Union[BaseModelBackend,
                                       List[BaseModelBackend],
                                       ModelManager]] = None,
                 agent_graph: "AgentGraph" = None,
                 available_actions: list[ActionType] = None,
                 tools: Optional[List[Union[FunctionTool, Callable]]] = None,
                 max_iteration: int = 1,
                 interview_record: bool = False,
                 token_limit: int | None = None,
                 message_window_size: int | None = None,
                 max_content_length: int = 280):
        self.social_agent_id = agent_id
        self.user_info = user_info
        self.channel = channel or Channel()
        self.locale = getattr(user_info, "locale", "en")
        self.platform_name = platform_display_name(
            getattr(user_info, "platform_name", None),
            self.locale)
        self.action = SocialAction(
            agent_id,
            self.channel,
            locale=self.locale,
            max_content_length=max_content_length,
        )
        self.env = SocialEnvironment(
            self.action, self.user_info, locale=self.locale)

        if user_info_template is None:
            system_message_content = self.user_info.to_system_message()
        else:
            system_message_content = self.user_info.to_custom_system_message(
                user_info_template)
        system_message = BaseMessage.make_assistant_message(
            role_name="system",
            content=system_message_content,  # system prompt
        )

        if not available_actions:
            agent_log.debug("No available actions defined, using all actions.")
            self.action_tools = self.env.action.get_function_tool_list_for_actions()
        else:
            all_tools = self.env.action.get_openai_function_list()
            all_possible_actions = [self._tool_name(tool) for tool in all_tools]

            for action in available_actions:
                action_name = action.value if isinstance(
                    action, ActionType) else action
                if action_name not in all_possible_actions:
                    agent_log.warning(
                        f"Action {action_name} is not supported. Supported "
                        f"actions are: {', '.join(all_possible_actions)}")
            self.action_tools = self.env.action.get_function_tool_list_for_actions(
                available_actions)
        all_tools = (tools or []) + (self.action_tools or [])
        super().__init__(
            system_message=system_message,
            model=model,
            scheduling_strategy='random_model',
            tools=all_tools,
            max_iteration=max_iteration,
            token_limit=token_limit,
            message_window_size=message_window_size,
            summarize_threshold=None,
        )
        if token_limit is not None:
            self.memory = ChatHistoryMemory(
                RecentTokenLimitContextCreator(
                    self.model_backend.token_counter,
                    token_limit,
                ),
                window_size=message_window_size,
                agent_id=self.agent_id,
            )
        self.max_iteration = max_iteration
        self.interview_record = interview_record
        self.agent_graph = agent_graph
        self.test_prompt = (
            "\n"
            "海伦是一位成功的作家，通常创作受欢迎的西部小说。现在，她有了一个可能产生"
            "很大影响的新小说想法。如果成功，她的事业会大幅提升；但如果失败，"
            "她会白白投入大量时间和精力。\n\n"
            "你认为海伦应该怎么做？"
        ) if is_zh_locale(self.locale) else (
            "\n"
            "Helen is a successful writer who usually writes popular western "
            "novels. Now, she has an idea for a new novel that could really "
            "make a big impact. If it works out, it could greatly "
            "improve her career. But if it fails, she will have spent "
            "a lot of time and effort for nothing.\n"
            "\n"
            "What do you think Helen should do?")

    def _system_message_without_response_method(self) -> str:
        content = self.system_message.content
        for marker in ("# RESPONSE METHOD", "# RESPONSE FORMAT", "# 回复方式"):
            content = content.split(marker)[0]
        return content

    def _platform_user_prompt(self) -> str:
        if is_zh_locale(self.locale):
            return f"你是一名{self.platform_name}用户。"
        return "You are a twitter user."

    async def perform_action_by_llm(self):
        # Get posts:
        env_prompt = await self.env.to_text_prompt()
        if is_zh_locale(self.locale):
            user_content = (
                "请在观察平台环境后执行一个社交媒体行动。注意，不要把行动局限为"
                f"点赞帖子。请结合你看到的帖子与个人画像行动。这是你的社交媒体环境：{env_prompt}")
        else:
            user_content = (
                f"Please perform social media actions after observing the "
                f"platform environments. Notice that don't limit your "
                f"actions for example to just like the posts. "
                f"Use the observed posts and your profile when acting. "
                f"Here is your social media environment: {env_prompt}")
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=user_content)
        try:
            response = await self.astep(user_msg)
            response_info = getattr(response, "info", {}) or {}
            tool_calls = (
                response_info.get("tool_calls")
                if isinstance(response_info, dict)
                else None
            ) or []
            if not tool_calls:
                return {
                    "success": False,
                    "error": "LLM response did not include a tool call.",
                }
            for tool_call in tool_calls:
                action_name = self._tool_call_name(tool_call)
                args = self._tool_call_args(tool_call)
                self._log_action_summary(action_name, args=args)
                return response
        except Exception as e:
            agent_log.error(f"Agent {self.social_agent_id} error: {e}")
            return e


    async def perform_test(self):
        """
        doing group polarization test for all agents.
        TODO: rewrite the function according to the ChatAgent.
        TODO: unify the test and interview function.
        """
        # user conduct test to agent
        _ = BaseMessage.make_user_message(
            role_name="User",
            content=self._platform_user_prompt())
        # Test memory should not be writed to memory.
        # self.memory.write_record(MemoryRecord(user_msg,
        #                                       OpenAIBackendRole.USER))

        openai_messages, num_tokens = self.memory.get_context()

        openai_messages = ([{
            "role":
            self.system_message.role_name,
            "content":
            self._system_message_without_response_method(),
        }] + openai_messages + [{
            "role": "user",
            "content": self.test_prompt
        }])

        self._log_action_summary("test")
        # NOTE: this is a temporary solution.
        # Camel can not stop updating the agents' memory after stop and astep
        # now.
        response = await self._aget_model_response(
            openai_messages=openai_messages, num_tokens=num_tokens)
        content = response.output_messages[0].content
        return {
            "user_id": self.social_agent_id,
            "prompt": openai_messages,
            "content": content
        }

    async def perform_interview(self, interview_prompt: str):
        """
        Perform an interview with the agent.
        """
        # user conduct test to agent
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=self._platform_user_prompt())

        if self.interview_record:
            # Test memory should not be writed to memory.
            self.update_memory(message=user_msg, role=OpenAIBackendRole.SYSTEM)

        openai_messages, num_tokens = self.memory.get_context()

        openai_messages = ([{
            "role":
            self.system_message.role_name,
            "content":
            self._system_message_without_response_method(),
        }] + openai_messages + [{
            "role": "user",
            "content": interview_prompt
        }])

        self._log_action_summary(
            ActionType.INTERVIEW.value,
            args={"prompt": interview_prompt})
        # NOTE: this is a temporary solution.
        # Camel can not stop updating the agents' memory after stop and astep
        # now.

        response = await self._aget_model_response(
            openai_messages=openai_messages, num_tokens=num_tokens)

        content = response.output_messages[0].content

        if self.interview_record:
            # Test memory should not be writed to memory.
            self.update_memory(message=response.output_messages[0],
                               role=OpenAIBackendRole.USER)
        # Record the complete interview (prompt + response) through the channel
        interview_data = {"prompt": interview_prompt, "response": content}
        result = await self.env.action.perform_action(
            interview_data, ActionType.INTERVIEW.value)

        # Return the combined result
        return {
            "user_id": self.social_agent_id,
            "prompt": openai_messages,
            "content": content,
            "success": result.get("success", False)
        }

    async def perform_action_by_hci(self) -> Any:
        print("请选择一个要执行的函数:" if is_zh_locale(self.locale)
              else "Please choose one function to perform:")
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            function_name = self._tool_name(function_list[i])
            label = zh_action_label(function_name) if is_zh_locale(
                self.locale) else function_name
            print(f"{i}: {label}")

        selection = int(input("请输入你的选择: " if is_zh_locale(self.locale)
                              else "Enter your choice: "))
        if not 0 <= selection < len(function_list):
            agent_log.error(f"Agent {self.social_agent_id} invalid input.")
            return
        func_name = self._tool_name(function_list[selection])
        func = getattr(self.action, func_name)

        params = inspect.signature(func).parameters
        args = []
        for param in params.values():
            while True:
                try:
                    value = input(
                        f"请输入 {param.name} 的值: "
                        if is_zh_locale(self.locale)
                        else f"Enter value for {param.name}: ")
                    args.append(value)
                    break
                except ValueError:
                    agent_log.error(
                        "输入无效，请输入整数。"
                        if is_zh_locale(self.locale)
                        else "Invalid input, please enter an integer.")

        result = await func(*args)
        return result

    async def perform_action_by_data(self, func_name, *args, **kwargs) -> Any:
        func_name = func_name.value if isinstance(func_name,
                                                  ActionType) else func_name
        func = getattr(self.action, func_name, None)
        if func is not None:
            result = await func(*args, **kwargs)
            if is_zh_locale(self.locale):
                action_memory = (
                    f"智能体 {self.social_agent_id} 执行了"
                    f"“{zh_action_label(func_name)}”，参数 args: {args}，"
                    f"kwargs: {kwargs}，结果为 {result}")
            else:
                action_memory = (
                    f"Agent {self.social_agent_id} performed "
                    f"{func_name} with args: {args} and kwargs: {kwargs}"
                    f"and the result is {result}")
            self.update_memory(message=BaseMessage.make_user_message(
                role_name=OpenAIBackendRole.SYSTEM,
                content=action_memory),
                               role=OpenAIBackendRole.SYSTEM)
            self._log_action_summary(func_name, args=args, kwargs=kwargs)
            return result
        raise ValueError(f"Function {func_name} not found in the list.")

    def perform_agent_graph_action(
        self,
        action_name: str,
        arguments: dict[str, Any],
    ):
        r"""Remove edge if action is unfollow or add edge
        if action is follow to the agent graph.
        """
        if "unfollow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.remove_edge(self.social_agent_id, followee_id)
            self._log_action_summary(
                action_name, kwargs={"followee_id": followee_id})
        elif "follow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.add_edge(self.social_agent_id, followee_id)
            self._log_action_summary(
                action_name, kwargs={"followee_id": followee_id})

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(agent_id={self.social_agent_id}, "
                f"model_type={self.model_type.value})")
