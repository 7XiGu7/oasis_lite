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
from dataclasses import dataclass, replace
from enum import Enum


class ActionType(Enum):
    EXIT = "exit"
    REFRESH = "refresh"
    SEARCH_USER = "search_user"
    SEARCH_POSTS = "search_posts"
    CREATE_POST = "create_post"
    LIKE_POST = "like_post"
    UNLIKE_POST = "unlike_post"
    DISLIKE_POST = "dislike_post"
    UNDO_DISLIKE_POST = "undo_dislike_post"
    REPORT_POST = "report_post"
    FOLLOW = "follow"
    UNFOLLOW = "unfollow"
    MUTE = "mute"
    UNMUTE = "unmute"
    TREND = "trend"
    SIGNUP = "sign_up"
    REPOST = "repost"
    QUOTE_POST = "quote_post"
    UPDATE_REC_TABLE = "update_rec_table"
    CREATE_COMMENT = "create_comment"
    LIKE_COMMENT = "like_comment"
    UNLIKE_COMMENT = "unlike_comment"
    DISLIKE_COMMENT = "dislike_comment"
    UNDO_DISLIKE_COMMENT = "undo_dislike_comment"
    DO_NOTHING = "do_nothing"
    PURCHASE_PRODUCT = "purchase_product"
    MENTION = "mention"
    INTERVIEW = "interview"
    JOIN_GROUP = "join_group"
    LEAVE_GROUP = "leave_group"
    SEND_TO_GROUP = "send_to_group"
    CREATE_GROUP = "create_group"
    LISTEN_FROM_GROUP = "listen_from_group"
    VIEW_COMMENT = "view_comment"

    @classmethod
    def get_default_twitter_actions(cls):
        return [
            cls.CREATE_POST,
            cls.LIKE_POST,
            cls.REPOST,
            cls.FOLLOW,
            cls.DO_NOTHING,
            cls.QUOTE_POST,
            cls.CREATE_COMMENT,
            # cls.LIKE_COMMENT,
        ]

    @classmethod
    def get_default_twitter_robot_actions(cls):
        return [
            cls.CREATE_POST,
            cls.LIKE_POST,
            cls.REPOST,
            cls.FOLLOW,
            cls.DO_NOTHING,
            cls.QUOTE_POST,
            cls.CREATE_COMMENT,
            cls.LIKE_COMMENT,
        ]

    @classmethod
    def get_default_weibo_actions(cls):
        return cls.get_default_twitter_actions()

    @classmethod
    def get_default_weibo_robot_actions(cls):
        return cls.get_default_twitter_robot_actions()

    @classmethod
    def get_default_reddit_actions(cls):
        return [
            cls.LIKE_POST,
            cls.DISLIKE_POST,
            cls.CREATE_POST,
            cls.CREATE_COMMENT,
            cls.LIKE_COMMENT,
            cls.DISLIKE_COMMENT,
            cls.SEARCH_POSTS,
            cls.SEARCH_USER,
            cls.TREND,
            cls.REFRESH,
            cls.DO_NOTHING,
            cls.FOLLOW,
            cls.MUTE,
        ]


class RecsysType(Enum):
    TWITTER = "twitter"
    WEIBO = "weibo"
    TWHIN = "twhin-bert"
    REDDIT = "reddit"
    RANDOM = "random"


@dataclass(frozen=True)
class RecsysConfig:
    recsys_type: str | RecsysType = RecsysType.REDDIT
    available_device: str = "cpu"
    refresh_rec_post_count: int = 1
    max_rec_post_len: int = 2
    following_post_count: int = 3
    time_top_n_comments: int | None = None
    use_openai_embedding: bool = False
    model_name_or_path: str = "Twitter/twhin-bert-base"
    embedding_batch_size: int | None = None
    openai_embedding_batch_size: int | None = None
    coarse_filter_size: int = 4000

    @classmethod
    def twitter(cls, available_device: str = "cpu", **overrides) -> "RecsysConfig":
        return cls(
            recsys_type=RecsysType.TWHIN,
            available_device=available_device,
            refresh_rec_post_count=3,
            max_rec_post_len=5,
            following_post_count=2,
            time_top_n_comments=5,
            **overrides,
        ).validated()

    @classmethod
    def weibo(cls, available_device: str = "cpu", **overrides) -> "RecsysConfig":
        return cls.twitter(available_device=available_device, **overrides)

    @classmethod
    def reddit(cls, available_device: str = "cpu", **overrides) -> "RecsysConfig":
        return cls(
            recsys_type=RecsysType.REDDIT,
            available_device=available_device,
            refresh_rec_post_count=5,
            max_rec_post_len=100,
            **overrides,
        ).validated()

    @property
    def recsys_kind(self) -> RecsysType:
        return RecsysType(self.recsys_type)

    def resolved_embedding_batch_size(self) -> int:
        if self.embedding_batch_size is not None:
            return self.embedding_batch_size
        return 256 if self.available_device.startswith("cuda") else 64

    def resolved_openai_embedding_batch_size(self) -> int:
        if self.openai_embedding_batch_size is not None:
            return self.openai_embedding_batch_size
        return 100

    def with_available_device(self, available_device: str) -> "RecsysConfig":
        return replace(self, available_device=available_device).validated()

    def validated(self) -> "RecsysConfig":
        RecsysType(self.recsys_type)
        if self.refresh_rec_post_count <= 0:
            raise ValueError("refresh_rec_post_count must be greater than 0.")
        if self.max_rec_post_len <= 0:
            raise ValueError("max_rec_post_len must be greater than 0.")
        if self.following_post_count < 0:
            raise ValueError("following_post_count must be non-negative.")
        if self.time_top_n_comments is not None and self.time_top_n_comments < 0:
            raise ValueError("time_top_n_comments must be non-negative.")
        if self.embedding_batch_size is not None and self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be greater than 0.")
        if (
            self.openai_embedding_batch_size is not None
            and self.openai_embedding_batch_size <= 0
        ):
            raise ValueError(
                "openai_embedding_batch_size must be greater than 0."
            )
        if self.coarse_filter_size <= 0:
            raise ValueError("coarse_filter_size must be greater than 0.")
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path must not be empty.")
        return self


class DefaultPlatformType(Enum):
    TWITTER = "twitter"
    WEIBO = "weibo"
    REDDIT = "reddit"
