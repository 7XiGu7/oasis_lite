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

import json
import sqlite3
from abc import ABC, abstractmethod
from string import Template

from oasis.social_agent.agent_action import SocialAction
from oasis.social_platform.database import get_db_path
from oasis.social_platform.typing import ActionType
from oasis.social_platform.config import UserInfo
from oasis.localization import is_zh_locale, zh_action_label


MAX_NOTIFICATIONS_PER_AGENT = 10


class Environment(ABC):

    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


class SocialEnvironment(Environment):
    followers_env_template = Template("I have $num_followers followers.")
    follows_env_template = Template("I have $num_follows follows.")

    posts_env_template = Template(
        "After refreshing, you see some posts:\n"
        "$posts\n"
    )

    groups_env_template = Template(
        "And there are many group chat channels $all_groups\n"
        "And You are already in some groups $joined_groups\n"
        "You receive some messages from them $messages\n"
        "You can join the groups you are interested, "
        "leave the groups you already in, send messages to the group "
        "you already in.\n"
        "You must make sure you can only send messages to the group you "
        "are already in")
    env_template = Template(
        # "$groups_env\n"
        "$posts_env\npick one you want to perform action that best "
        "reflects your current inclination based on your profile and "
        "posts content. Do not limit your action in just `like` to like posts")

    zh_followers_env_template = Template("我有 $num_followers 个粉丝。")
    zh_follows_env_template = Template("我关注了 $num_follows 个用户。")
    zh_posts_env_template = Template(
        "刷新后，你看到了以下帖子：\n"
        "$posts\n"
    )
    zh_groups_env_template = Template(
        "平台中有这些群聊频道：$all_groups\n"
        "你已经加入的群组是：$joined_groups\n"
        "你收到的群消息是：$messages\n"
        "你可以加入感兴趣的群组、退出已加入的群组，或向已加入的群组发送消息。\n"
        "你必须确保只能向自己已经加入的群组发送消息。")
    zh_env_template = Template(
        "$posts_env\n请选择一个最符合你当前倾向、个人画像和帖子内容的行动。"
        "不要只局限于点赞，也可以根据情况选择其他行动。")

    def __init__(self,
                 action: SocialAction,
                 user_info: UserInfo | None = None,
                 locale: str | None = None):
        self.action = action
        self.user_info = user_info
        self.locale = locale or getattr(user_info, "locale", "en")
        if is_zh_locale(self.locale):
            self.followers_env_template = self.zh_followers_env_template
            self.follows_env_template = self.zh_follows_env_template
            self.posts_env_template = self.zh_posts_env_template
            self.groups_env_template = self.zh_groups_env_template
            self.env_template = self.zh_env_template
        self.posts = None
        self.notifications = None
        self.current_time = 0

    async def get_posts(self):
        if self.posts is None:
            posts = await self.action.refresh()
            if posts["success"]:
                self.current_time = int(posts["current_time"])
                posts = posts['posts']
                # 确保 post_attr 是字典类型
                for post in posts:
                    post_attr = post.get('post_attr')
                    if isinstance(post_attr, str):
                        post_attr = json.loads(post_attr) if post_attr else {}
                    elif post_attr is None:
                        post_attr = {}
                    post['post_attr'] = post_attr
                self.posts = posts

                return self.posts
            else:
                raise ValueError('Failed to get posts: {}'.format(posts.get("error", "Unknown error")))
        else:
            return self.posts

    async def get_notifications(self):
        agent_id = self.action.agent_id
        db_path = get_db_path()
        notifications = []
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT interaction_id, user_id, post_id, comment_id, "
                "interaction_type FROM interaction WHERE original_user_id = ? "
                "ORDER BY created_at DESC, interaction_id DESC LIMIT ?",
                (agent_id, MAX_NOTIFICATIONS_PER_AGENT),
            )
            interaction_ids = []
            for row in cursor.fetchall():
                (interaction_id, user_id, post_id, comment_id, interaction_type) = row
                interaction_ids.append(interaction_id)
                cursor.execute("SELECT name FROM user WHERE user_id = ?",
                               (user_id,))
                user_name = cursor.fetchone()[0]
                #区分出需要显示内容的通知类型
                if interaction_type in [ActionType.CREATE_COMMENT.value,ActionType.QUOTE_POST.value]:
                    if interaction_type == ActionType.CREATE_COMMENT.value:
                        cursor.execute("SELECT content FROM comment WHERE comment_id = ?",
                                       (comment_id,))
                        comment_result = cursor.fetchone()
                        content = comment_result[0] if comment_result else ""
                        if is_zh_locale(self.locale):
                            notification_content = f"{user_name} 评论了 {self.user_info.name} 的帖子，帖子 ID：{post_id}，评论内容：“{content}”"
                        else:
                            notification_content = f"{user_name} commented on {self.user_info.name}'s post, post id: {post_id}, Comment Content: \"{content}\""
                    else:
                        cursor.execute("SELECT quote_content FROM post WHERE post_id = ?",
                                       (post_id,))
                        post_result = cursor.fetchone()
                        content = post_result[0] if post_result else ""
                        if is_zh_locale(self.locale):
                            notification_content = f"{user_name} 引用了 {self.user_info.name} 的帖子，帖子 ID：{post_id}，引用内容：“{content}”"
                        else:
                            notification_content = f"{user_name} quoted {self.user_info.name}'s post, post id: {post_id}, Quote Content: \"{content}\""
                else:
                    if interaction_type == ActionType.FOLLOW.value:
                        if is_zh_locale(self.locale):
                            notification_content = f"{user_name} 对 {self.user_info.name} 执行了“{zh_action_label(interaction_type)}”行动。"
                        else:
                            notification_content = f"{user_name} performed '{interaction_type}' action on {self.user_info.name}."
                    else:
                        if is_zh_locale(self.locale):
                            notification_content = f"{user_name} 对 {self.user_info.name} 的帖子执行了“{zh_action_label(interaction_type)}”行动，帖子 ID：{post_id}。"
                        else:
                            notification_content = f"{user_name} performed '{interaction_type}' action on {self.user_info.name}'s post, post id: {post_id}."
                notifications.append({
                    "user_id": user_id,
                    "user_name": user_name,
                    "post_id": post_id,
                    "content": notification_content,
                })
            # 删除已读取的 interaction 记录
            if interaction_ids:
                placeholders = ",".join("?" for _ in interaction_ids)
                cursor.execute(f"DELETE FROM interaction WHERE interaction_id IN ({placeholders})",
                               tuple(interaction_ids))
                conn.commit()
            conn.close()
        except Exception as e:
            print(f"Failed to get notifications, error:{e}")
        self.notifications = notifications
        return notifications


    async def get_posts_env(self) -> str:
        posts = await self.action.refresh()#这个方法不能连续调用两次，否则无法将行动信息插入Trcae表，因为是相同的记录
        # TODO: Replace posts json format string to other formats
        if posts["success"]:
            posts_env = json.dumps(posts["posts"], ensure_ascii=False)
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "刷新后，没有看到任何已有帖子。" if is_zh_locale(self.locale) else "After refreshing, there are no existing posts."
        return posts_env

    def clear_env(self):
        self.posts = None
        self.notifications = None


    async def get_followers_env(self) -> str:
        # TODO: Implement followers env
        agent_id = self.action.agent_id
        db_path = get_db_path()
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT num_followers FROM user WHERE agent_id = ?",
                           (agent_id, ))
            result = cursor.fetchone()
            num_followers = result[0] if result else 0
            conn.close()
        except Exception:
            num_followers = 0
        return self.followers_env_template.substitute(
            {"num_followers": num_followers})

    async def get_follows_env(self) -> str:
        # TODO: Implement follows env
        agent_id = self.action.agent_id
        try:
            db_path = get_db_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT num_followings FROM user WHERE agent_id = ?",
                (agent_id, ))
            result = cursor.fetchone()
            num_followings = result[0] if result else 0
            conn.close()
        except Exception:
            num_followings = 0
        return self.follows_env_template.substitute(
            {"num_follows": num_followings})

    async def get_group_env(self) -> str:
        groups = await self.action.listen_from_group()
        if groups["success"]:
            all_groups = json.dumps(groups["all_groups"], ensure_ascii=False)
            joined_groups = json.dumps(groups["joined_groups"], ensure_ascii=False)
            messages = json.dumps(groups["messages"], ensure_ascii=False)
            groups_env = self.groups_env_template.substitute(
                all_groups=all_groups,
                joined_groups=joined_groups,
                messages=messages,
            )
        else:
            groups_env = "没有群组。" if is_zh_locale(self.locale) else "No groups."
        return groups_env

    async def to_text_prompt(
        self,
        include_posts: bool = True,
        include_followers: bool = True,
        include_follows: bool = True,
    ) -> str:
        no_followers = "没有粉丝。" if is_zh_locale(self.locale) else "No followers."
        no_follows = "没有关注任何用户。" if is_zh_locale(self.locale) else "No follows."
        followers_env = (await self.get_followers_env()
                         if include_follows else no_followers)
        follows_env = (await self.get_follows_env()
                       if include_followers else no_follows)
        posts_env = await self.get_posts_env() if include_posts else ""

        return self.env_template.substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
            # groups_env=await self.get_group_env(),
        )


    # 获取多条帖子作者的帖子信息
    async def get_users_posts(self, user_ids:list) -> dict:
        try:
            db_path = get_db_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            results = {user_id: [] for user_id in user_ids}
            for user_id in user_ids:
                post_query = (
                    f"SELECT post_id, user_id, original_post_id, content, "
                    f"quote_content, created_at, num_likes, num_dislikes, "
                    f"num_shares FROM post WHERE user_id = ?"
                    f"ORDER BY created_at DESC LIMIT 5")
                cursor.execute(post_query, (user_id,))
                for row in cursor.fetchall():
                    (post_id, user_id, original_post_id, content, quote_content,
                     created_at, num_likes, num_dislikes, num_shares) = row
                    results[user_id].append({
                        "post_id":
                            post_id,
                        "user_id":
                            user_id,
                        "content":
                            content,
                        "created_at":
                            created_at,
                        "num_likes":
                            num_likes,
                        "num_shares":
                            num_shares,
                    })
            conn.close()
        except Exception as e:
            print('error in get_users_posts:', e)
            results = {user_id: [] for user_id in user_ids}
        return results

    # 获取多条帖子作者的粉丝量
    async def get_users_follower_nums(self, user_ids: list) -> dict:
        try:
            db_path = get_db_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            results = {user_id: 0 for user_id in user_ids}
            for user_id in user_ids:
                query = (
                    f"SELECT num_followers FROM user WHERE user_id = ?")
                cursor.execute(query, (user_id,))
                for row in cursor.fetchall():
                    (num_followers,) = row
                    results[user_id]= num_followers
            conn.close()
        except Exception as e:
            print('error in get_users_follower_nums:', e)
            results = {user_id: 0 for user_id in user_ids}
        return results

    # 获取多个帖子作者和当前用户的共同关注数
    async def get_users_co_following_nums(self, user_ids: list) -> tuple[dict, dict, list]:
        current_user_id = self.action.agent_id
        user_ids.append(current_user_id)
        try:
            db_path = get_db_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            results = {user_id: [] for user_id in user_ids}
            for user_id in user_ids:
                post_query = (
                    f"SELECT followee_id FROM follow WHERE follower_id = ?")
                cursor.execute(post_query, (user_id,))
                for row in cursor.fetchall():
                    (followee_id,) = row
                    results[user_id].append(followee_id)
            conn.close()
        except Exception as e:
            print('error in get_users_co_following_nums:', e)
            results = {user_id: [] for user_id in user_ids}

        # 判断当前用户的关注列表里是否已经存在帖子作者
        current_followees = results[current_user_id]
        filtered_user_ids = {user_id: False for user_id in user_ids[:-1]}
        for user_id in user_ids[:-1]:
            if user_id in current_followees:
                filtered_user_ids[user_id] = True
        # 计算每个帖子作者和当前用户的共同关注数
        common_followee_nums = {user_id: 0 for user_id in user_ids[:-1] if not filtered_user_ids[user_id]}
        for user_id in user_ids[:-1]:
            common_followee_nums[user_id] = len(set(results[user_id]).intersection(set(current_followees)))
        #判断当前用户是否在其余帖子作者的关注列表里
        is_follows = {user_id: False for user_id in user_ids[:-1] if not filtered_user_ids[user_id]}
        for user_id in user_ids[:-1]:
            if current_user_id in results[user_id]:
                is_follows[user_id] = True

        updated_user_ids = list(is_follows.keys())

        return common_followee_nums, is_follows, updated_user_ids
