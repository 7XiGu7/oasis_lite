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
import asyncio
import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

from oasis.environment.env_action import LLMAction, ManualAction
try:
    from oasis.social_agent.llm_model import LLMSkippableError
except ModuleNotFoundError:
    class LLMSkippableError(RuntimeError):
        pass
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_agent.agents_generator import generate_custom_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import (ActionType, DefaultPlatformType,
                                          RecsysConfig, RecsysType)

# Create log directory if it doesn't exist
log_dir = os.getenv("OASIS_LOG_DIR", "outputs/logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logger
env_log = logging.getLogger("oasis.env")
env_log.setLevel("INFO")

# Add file handler to save logs to file
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
env_log_path = os.getenv(
    "OASIS_ENV_LOG_PATH",
    f"{log_dir}/oasis-{current_time}.log",
)
Path(env_log_path).parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(env_log_path, encoding="utf-8")
file_handler.setLevel("INFO")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
env_log.addHandler(file_handler)


class OasisEnv:

    def __init__(
        self,
        agent_graph: AgentGraph,
        platform: Union[DefaultPlatformType, Platform],
        database_path: str = None,
        semaphore: int = 128,
        max_content_length: int | None = None,
        recsys_config: RecsysConfig | None = None,
    ) -> None:
        r"""Init the oasis environment.

        Args:
            agent_graph: The AgentGraph to use in the simulation.
            platform: The platform type to use. Including
                `DefaultPlatformType.TWITTER` or `DefaultPlatformType.REDDIT`.
                Or you can pass a custom `Platform` instance.
            database_path: The path to create a sqlite3 database. The file
                extension must be `.db` such as `twitter_simulation.db`.
            agent_profile_path: The path to the agent profile. Make sure the
                data format is align with the `platform`.
            agent_models: The model backend to use for all agents to generate
                responses. (default: :obj:`ModelPlatformType.DEFAULT` with
                `ModelType.DEFAULT`)
            available_actions: The actions to use for the agents. Choose from
                `ActionType`.
        """
        # Initialize the agent graph
        self.agent_graph = agent_graph
        # Use a semaphore to limit the number of concurrent requests
        self.llm_semaphore = asyncio.Semaphore(semaphore)
        if isinstance(platform, DefaultPlatformType):
            if database_path is None:
                raise ValueError(
                    "database_path is required for DefaultPlatformType")
            self.platform = platform
            if platform == DefaultPlatformType.TWITTER:
                self.channel = Channel()
                platform_recsys_config = (
                    recsys_config
                    if recsys_config is not None
                    else RecsysConfig.twitter()
                )
                self.platform = Platform(
                    db_path=database_path,
                    recsys_config=platform_recsys_config,
                    channel=self.channel,
                    max_content_length=max_content_length,
                )
                self.platform_type = DefaultPlatformType.TWITTER
            elif platform == DefaultPlatformType.WEIBO:
                self.channel = Channel()
                platform_recsys_config = (
                    recsys_config
                    if recsys_config is not None
                    else RecsysConfig.weibo()
                )
                self.platform = Platform(
                    db_path=database_path,
                    recsys_config=platform_recsys_config,
                    channel=self.channel,
                    locale="zh_cn",
                    platform_name="微博",
                    max_content_length=max_content_length,
                )
                self.platform_type = DefaultPlatformType.WEIBO
            elif platform == DefaultPlatformType.REDDIT:
                self.channel = Channel()
                platform_recsys_config = (
                    recsys_config
                    if recsys_config is not None
                    else RecsysConfig.reddit()
                )
                self.platform = Platform(
                    db_path=database_path,
                    channel=self.channel,
                    recsys_config=platform_recsys_config,
                    allow_self_rating=True,
                    show_score=True,
                )
                self.platform_type = DefaultPlatformType.REDDIT
            else:
                raise ValueError(f"Invalid platform: {platform}. Only "
                                 "DefaultPlatformType.TWITTER, "
                                 "DefaultPlatformType.WEIBO or "
                                 "DefaultPlatformType.REDDIT are supported.")
        elif isinstance(platform, Platform):
            if database_path != platform.db_path:
                env_log.warning("database_path is not the same as the "
                                "platform.db_path, using the platform.db_path")
            self.platform = platform
            self.channel = platform.channel
            if platform.recsys_type == RecsysType.REDDIT:
                self.platform_type = DefaultPlatformType.REDDIT
            elif getattr(platform, "locale", "en") == "zh_cn" or platform.recsys_type == RecsysType.WEIBO:
                self.platform_type = DefaultPlatformType.WEIBO
            else:
                self.platform_type = DefaultPlatformType.TWITTER
        else:
            raise ValueError(
                f"Invalid platform: {platform}. You should pass a "
                "DefaultPlatformType or a Platform instance.")

    def _connect_agents_to_channel(self) -> None:
        if self.agent_graph is None:
            return
        for _, agent in self.agent_graph.get_agents():
            agent.channel = self.channel
            agent.env.action.channel = self.channel

    async def reset(self, initialize_social_state: bool = True) -> None:
        r"""Start the platform and sign up the agents as well as 初始化社交关系."""
        self.platform_task = asyncio.create_task(self.platform.running())
        if not initialize_social_state:
            self._connect_agents_to_channel()
            return
        if self.platform_type in (
            DefaultPlatformType.TWITTER,
            DefaultPlatformType.WEIBO,
        ):
            self.agent_graph = await self._generate_custom_agents_batch()
        else:
            self.agent_graph = await generate_custom_agents(
                channel=self.channel, agent_graph=self.agent_graph)

    async def _generate_custom_agents_batch(self) -> AgentGraph:
        if self.agent_graph is None:
            self.agent_graph = AgentGraph()

        for _, agent in self.agent_graph.get_agents():
            agent.channel = self.channel
            agent.env.action.channel = self.channel

        agents = list(self.agent_graph.get_agents())
        current_time = self.platform.sandbox_clock.get_time_step()
        pl_utils = self.platform.pl_utils

        user_rows = []
        signup_trace_rows = []
        for agent_id, agent in agents:
            user_rows.append((
                agent_id,
                agent_id,
                agent.user_info.user_name,
                agent.user_info.name,
                agent.user_info.description,
                current_time,
                0,
                0,
            ))
            action_info = {
                "name": agent.user_info.name,
                "user_name": agent.user_info.user_name,
                "bio": agent.user_info.description,
            }
            signup_trace_rows.append((
                agent_id,
                current_time,
                ActionType.SIGNUP.value,
                json.dumps(action_info),
            ))

        pl_utils._execute_many_db_command(
            "INSERT INTO user (user_id, agent_id, user_name, name, bio, "
            "created_at, num_followings, num_followers) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?)",
            user_rows,
            commit=True,
        )
        pl_utils._execute_many_db_command(
            "INSERT INTO trace (user_id, created_at, action, info) "
            "VALUES (?, ?, ?, ?)",
            signup_trace_rows,
            commit=True,
        )

        create_post_tasks = [
            agent.env.action.create_post(
                content=agent.user_info.profile["other_info"][
                    "previous_tweets"][0])
            for _, agent in agents
            if agent.user_info.profile["other_info"]["previous_tweets"]
        ]
        await asyncio.gather(*create_post_tasks)

        follow_edges: list[tuple[int, int]] = []
        seen_edges: set[tuple[int, int]] = set()
        for agent_id, agent in agents:
            following_agent_ids = agent.user_info.profile["other_info"][
                "following_id_list"]
            for followee_id in following_agent_ids:
                edge = (agent_id, followee_id)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                follow_edges.append(edge)

        if follow_edges:
            pl_utils._execute_db_command(
                "SELECT COALESCE(MAX(follow_id), 0) FROM follow")
            start_follow_id = pl_utils.db_cursor.fetchone()[0] + 1
            follow_rows = [
                (
                    start_follow_id + index,
                    follower_id,
                    followee_id,
                    current_time,
                )
                for index, (follower_id, followee_id) in enumerate(follow_edges)
            ]
            follow_trace_rows = [
                (
                    follower_id,
                    current_time,
                    ActionType.FOLLOW.value,
                    json.dumps({"follow_id": follow_id}),
                )
                for follow_id, follower_id, _, _ in follow_rows
            ]
            # interaction_rows = [
            #     (
            #         follower_id,
            #         None,
            #         None,
            #         followee_id,
            #         0,
            #         ActionType.FOLLOW.value,
            #         current_time,
            #     )
            #     for _, follower_id, followee_id, _ in follow_rows
            # ]

            pl_utils._execute_many_db_command(
                "INSERT INTO follow (follow_id, follower_id, followee_id, "
                "created_at) VALUES (?, ?, ?, ?)",
                follow_rows,
                commit=True,
            )
            pl_utils._execute_many_db_command(
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)",
                follow_trace_rows,
                commit=True,
            )
            # pl_utils._execute_many_db_command(
            #     "INSERT INTO interaction (user_id, post_id, comment_id, "
            #     "original_user_id, is_mention, interaction_type, created_at) "
            #     "VALUES (?, ?, ?, ?, ?, ?, ?)",
            #     interaction_rows,
            #     commit=True,
            # )

            following_counts = Counter(
                follower_id for follower_id, _ in follow_edges)
            follower_counts = Counter(
                followee_id for _, followee_id in follow_edges)
            pl_utils._execute_many_db_command(
                "UPDATE user SET num_followings = num_followings + ? "
                "WHERE user_id = ?",
                [
                    (count, user_id)
                    for user_id, count in following_counts.items()
                ],
                commit=True,
            )
            pl_utils._execute_many_db_command(
                "UPDATE user SET num_followers = num_followers + ? "
                "WHERE user_id = ?",
                [
                    (count, user_id)
                    for user_id, count in follower_counts.items()
                ],
                commit=True,
            )

        return self.agent_graph


    async def _perform_llm_action(self, agent):
        r"""Send the request to the llm model and execute the action.
        """
        try:
            async with self.llm_semaphore:
                return await agent.perform_action_by_llm()
                # return await agent.perform_action_by_behavior_model()
        except LLMSkippableError as e:
            agent_id = getattr(agent, "social_agent_id", "unknown")
            env_log.warning(
                "[LLM_SKIPPED] 跳过当前 agent 的 LLM 行为；"
                "agent_id=%s, reason=%s",
                agent_id,
                e,
            )
            return None

    async def _perform_interview_action(self, agent, interview_prompt: str):
        r"""Send the request to the llm model and execute the interview.
        """
        try:
            async with self.llm_semaphore:
                return await agent.perform_interview(interview_prompt)
        except LLMSkippableError as e:
            agent_id = getattr(agent, "social_agent_id", "unknown")
            env_log.warning(
                "[LLM_SKIPPED] 跳过当前 agent 的访谈行为；"
                "agent_id=%s, reason=%s",
                agent_id,
                e,
            )
            return None

    async def step(
        self, actions: dict[Any, Union[ManualAction, LLMAction,
                                               List[Union[ManualAction,
                                                          LLMAction]]]]
    ) -> list[Any]:
        r"""Update the recommendation system and perform the actions.

        Args:
            actions(dict[Any, Union[ManualAction, LLMAction,
                List[Union[ManualAction, LLMAction]]]]): The actions to
                perform, including the manual(pre-defined) actions and llm
                actions.
        Returns:
            Per-action task results.
        """

        # Update the recommendation system
        await self.platform.update_rec_table()
        env_log.info("更新推荐系统完毕.")

        # Create tasks for both manual and LLM actions
        tasks = []
        for agent, action in actions.items():
            #如果，智能体不是社交机器人，且行动不是人工行动，则概率激活并执行行动
            if isinstance(action, list):
                for single_action in action:
                    if isinstance(single_action, ManualAction):
                        if single_action.action_type == ActionType.INTERVIEW:
                            # Use the agent's perform_interview method for
                            # interview actions
                            interview_prompt = single_action.action_args.get(
                                "prompt", "")
                            tasks.append(
                                self._perform_interview_action(
                                    agent, interview_prompt))
                        else:
                            tasks.append(
                                agent.perform_action_by_data(
                                    single_action.action_type,
                                    **single_action.action_args))
                    elif isinstance(single_action, LLMAction):
                        tasks.append(self._perform_llm_action(agent))
            else:
                if isinstance(action, ManualAction):
                    if action.action_type == ActionType.INTERVIEW:
                        # Use the agent's perform_interview method for
                        # interview actions
                        interview_prompt = action.action_args.get("prompt", "")
                        tasks.append(
                            self._perform_interview_action(
                                agent, interview_prompt))
                    else:
                        tasks.append(
                            agent.perform_action_by_data(
                                action.action_type, **action.action_args))
                elif isinstance(action, LLMAction):
                    tasks.append(self._perform_llm_action(agent))

        # Execute all tasks concurrently.
        results = await asyncio.gather(*tasks)
        env_log.info("performed all actions.")


        # Update the clock
        if self.platform_type in (DefaultPlatformType.TWITTER, DefaultPlatformType.WEIBO):
            self.platform.sandbox_clock.time_step += 1
        return results

    async def step_V2(
        self, actions: dict[Any, Union[ManualAction, LLMAction,
                                               List[Union[ManualAction,
                                                          LLMAction]]]]
    ) -> list[Any]:
        r"""Update the recommendation system and perform the actions.

        Args:
            actions(dict[Any, Union[ManualAction, LLMAction,
                List[Union[ManualAction, LLMAction]]]]): The actions to
                perform, including the manual(pre-defined) actions and llm
                actions.
        Returns:
            Per-action task results.
        """

        # Update the recommendation system
        await self.platform.update_rec_table()
        env_log.info("update rec table.")

        # Create tasks for both manual and LLM actions
        tasks = []
        for agent, action in actions.items():
            if isinstance(action, ManualAction):

                tasks.append(
                    agent.perform_action_by_data(
                        action.action_type, **action.action_args))
            elif isinstance(action, LLMAction):
                tasks.append(self._perform_llm_action(agent))

        # Execute all tasks concurrently.
        results = await asyncio.gather(*tasks)
        env_log.info("performed all actions.")


        # Update the clock
        if self.platform_type in (DefaultPlatformType.TWITTER, DefaultPlatformType.WEIBO):
            self.platform.sandbox_clock.time_step += 1
        return results

    async def close(self) -> None:
        r"""Stop the platform and close the environment.
        """
        platform_task = getattr(self, "platform_task", None)
        if platform_task is None:
            return
        if platform_task.done():
            await platform_task
            return
        await self.channel.write_to_receive_queue(
            (None, None, ActionType.EXIT))
        await platform_task
        env_log.info("Simulation finished! Please check the results in the "
                     f"database: {self.platform.db_path}. Note that the trace "
                     "table stored all the actions of the agents.")
