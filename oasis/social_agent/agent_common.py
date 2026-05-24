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

import logging
import os
from datetime import datetime

from oasis.social_platform.typing import ActionType

log_dir = "/data/lijiantong/Data/oasis/log"
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
AGENT_ACTION_LOG_PATH = os.getenv(
    "OASIS_AGENT_ACTION_RECORD_PATH",
    os.path.join(log_dir, f"agent_action_records_{str(now)}.jsonl"),
)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
os.makedirs(os.path.dirname(AGENT_ACTION_LOG_PATH), exist_ok=True)

# 创建独立的日志记录器
agent_log = logging.getLogger(name="social.agent")
agent_log.setLevel("DEBUG")

# 检查是否已添加处理器，避免重复添加
if not any(isinstance(handler, logging.FileHandler) for handler in agent_log.handlers):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_log_path = os.getenv(
        "OASIS_AGENT_LOG_PATH",
        f"{log_dir}/social.agent-{str(now)}.log",
    )
    os.makedirs(os.path.dirname(agent_log_path), exist_ok=True)
    file_handler = logging.FileHandler(
        agent_log_path, encoding='utf-8')
    file_handler.setLevel("DEBUG")
    file_handler.setFormatter(
        logging.Formatter(
            "%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
    agent_log.addHandler(file_handler)

ALL_SOCIAL_ACTIONS = [action.value for action in ActionType]


def _serialize_obj(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    if hasattr(obj, 'dict'):
        return obj.dict()
    return str(obj)
