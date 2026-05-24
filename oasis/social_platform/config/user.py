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
# flake8: noqa: E501
import warnings
from dataclasses import dataclass
from typing import Any

try:
    from camel.prompts import TextPrompt
except ModuleNotFoundError:
    class TextPrompt(str):
        @property
        def key_words(self):
            return set()
from oasis.localization import is_zh_locale, platform_display_name


@dataclass
class UserInfo:
    user_name: str | None = None
    name: str | None = None
    gender: str | None = None
    age: int | None = None
    description: str | None = None
    profile: dict[str, Any] | None = None
    recsys_type: str = "twitter"
    is_controllable: bool = False
    locale: str = "en"
    platform_name: str | None = None
    def to_custom_system_message(self, user_info_template: TextPrompt) -> str:
        required_keys = user_info_template.key_words
        info_keys = set(self.profile.keys())
        missing = required_keys - info_keys
        extra = info_keys - required_keys
        if missing:
            raise ValueError(
                f"Missing required keys in UserInfo.profile: {missing}")
        if extra:
            warnings.warn(f"Extra keys not used in UserInfo.profile: {extra}")

        return user_info_template.format(**self.profile)

    def to_system_message(self) -> str:
        if is_zh_locale(self.locale):
            return self.to_weibo_system_message()
        if self.recsys_type != "reddit":
            return self.to_twitter_system_message()
        else:
            return self.to_reddit_system_message()

    def to_weibo_system_message(self) -> str:
        platform = platform_display_name(self.platform_name, self.locale)
        other_info = self.profile.get("other_info", {}) if self.profile else {}
        traits_info = other_info.get("traits_info") or other_info.get("traits", "")
        user_profile = other_info.get("user_profile", self.description or "")

        return f"""
# 目标
你是 {platform} 上的一名真实用户。接下来我会向你展示一些帖子；看完后，请通过工具调用执行一个最符合你当前状态社交媒体行动。

# 个人画像
姓名：{self.name}
性别：{self.gender}
人格特征：
{traits_info}
描述：{user_profile}

# 回复方式
请通过工具调用执行行动，并且每次回复只调用一次工具。
        """


    def to_twitter_system_message(self) -> str:
        name_string = ""
        description_string = ""
        if self.name is not None:
            name_string = f"Your name is {self.name}."
        if self.profile is None:
            description = name_string
        elif "other_info" not in self.profile:
            description = name_string
        elif "user_profile" in self.profile["other_info"]:
            if self.profile["other_info"]["user_profile"] is not None:
                user_profile = self.profile["other_info"]["user_profile"]
                description_string = f"Your have profile: {user_profile}."
                description = f"{name_string}\n{description_string}"

        system_content = f"""
# OBJECTIVE
You're a Twitter user, and I'll present you with some posts. After you see the posts, choose an action from the available functions.

# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.
Name: {self.name}
Description: {self.description}
Personality traits: {self.profile['other_info']['traits']} "

# RESPONSE METHOD
Please perform actions by tool calling.
        """

        return system_content



    def to_reddit_system_message(self) -> str:
        name_string = ""
        description_string = ""
        if self.name is not None:
            name_string = f"Your name is {self.name}."
        if self.profile is None:
            description = name_string
        elif "other_info" not in self.profile:
            description = name_string
        elif "user_profile" in self.profile["other_info"]:
            if self.profile["other_info"]["user_profile"] is not None:
                user_profile = self.profile["other_info"]["user_profile"]
                description_string = f"Your have profile: {user_profile}."
                description = f"{name_string}\n{description_string}"
                print(self.profile['other_info'])
                description += (
                    f"You are a {self.profile['other_info']['gender']}, "
                    f"{self.profile['other_info']['age']} years old, with an MBTI "
                    f"personality type of {self.profile['other_info']['mbti']} from "
                    f"{self.profile['other_info']['country']}.")

        system_content = f"""
# OBJECTIVE
You're a Reddit user, and I'll present you with some tweets. After you see the tweets, choose some actions from the following functions.

# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.
{description}

# RESPONSE METHOD
Please perform actions by tool calling.
"""
        return system_content
