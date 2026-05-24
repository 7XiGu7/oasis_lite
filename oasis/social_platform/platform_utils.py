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
import json
from datetime import datetime

from oasis.social_platform.typing import RecsysType, ActionType
from oasis.localization import is_zh_locale


class PlatformUtils:

    def __init__(self,
                 db,
                 db_cursor,
                 start_time,
                 sandbox_clock,
                 show_score,
                 recsys_type,
                 report_threshold=1,
                 locale: str = "en"):
        self.db = db
        self.db_cursor = db_cursor
        self.start_time = start_time
        self.sandbox_clock = sandbox_clock
        self.show_score = show_score
        self.recsys_type = recsys_type
        self.report_threshold = report_threshold
        self.locale = locale

    @staticmethod
    def _not_signup_error_message(agent_id):
        return {
            "success":
                False,
            "error": (f"Agent {agent_id} has not signed up and does not have "
                      f"a user id."),
        }

    def _execute_db_command(self, command, args=(), commit=False):
        self.db_cursor.execute(command, args)
        if commit:
            self.db.commit()
        return self.db_cursor

    def _execute_many_db_command(self, command, args_list, commit=False):
        self.db_cursor.executemany(command, args_list)
        if commit:
            self.db.commit()
        return self.db_cursor

    def _check_agent_userid(self, agent_id):
        try:
            user_query = "SELECT user_id FROM user WHERE agent_id = ?"
            results = self._execute_db_command(user_query, (agent_id,))
            # Fetch the first row of the query result
            first_row = results.fetchone()
            if first_row:
                user_id = first_row[0]
                return user_id
            else:
                return None
        except Exception as e:
            # Log or handle the error as appropriate
            print(f"Error querying user_id for agent_id {agent_id}: {e}")
            return None

    def _add_comments_to_posts(self, posts_results, following_posts_ids, time_top_n_comments=None):
        # Initialize the returned posts list
        posts = []
        for row in posts_results:
            (post_id, user_id, original_post_id, content, quote_content,
             created_at, num_likes, num_dislikes, num_shares, post_attr) = row
            post_type_result = self._get_post_type(post_id)
            if post_type_result is None:
                continue
            original_user_id_query = (
                "SELECT user_id FROM post WHERE post_id = ?")

            num_reports = 0
            if post_type_result["type"] == "repost":
                self.db_cursor.execute(original_user_id_query,
                                       (original_post_id,))
                original_user_id = self.db_cursor.fetchone()[0]
                original_post_id = post_id
                post_id = post_type_result["root_post_id"]
                self.db_cursor.execute(
                    "SELECT content, quote_content, created_at, num_likes, "
                    "num_dislikes, num_shares, num_reports FROM post "
                    "WHERE post_id = ?", (post_id, ))
                original_post_result = self.db_cursor.fetchone()
                (content, quote_content, created_at, num_likes, num_dislikes,
                 num_shares, num_reports) = original_post_result
                user_names = self._get_user_names([user_id,original_user_id])
                post_content = (
                    f"{user_names[0]} reposted a post from "
                    f"{user_names[1]}. Repost content: {content}. ")

            elif post_type_result["type"] == "quote":
                self.db_cursor.execute(original_user_id_query,
                                       (original_post_id,))
                original_user_id = self.db_cursor.fetchone()[0]
                user_names = self._get_user_names([user_id, original_user_id])
                post_content = (
                    f"{user_names[0]} quoted a post from "
                    f"{user_names[1]}. Quote content: {quote_content}. "
                    f"Original Content: {content}")

            elif post_type_result["type"] == "common":
                user_name = self._get_user_names([user_id,])[0]
                post_content = f"{user_name} posted: {content}"
                # Get num_reports for common posts
                self.db_cursor.execute(
                    "SELECT num_reports FROM post WHERE post_id = ?",
                    (post_id, ))
                num_reports = self.db_cursor.fetchone()[0]

            # For each post, query its corresponding comments
            if time_top_n_comments is None:
                self.db_cursor.execute(
                    "SELECT comment_id, post_id, user_id, content, created_at, "
                    "num_likes, num_dislikes FROM comment WHERE post_id = ?",
                    (post_id,),
                )
            else:
                self.db_cursor.execute(
                    "SELECT comment_id, post_id, user_id, content, created_at, "
                    "num_likes, num_dislikes FROM comment WHERE post_id = ? "
                    "ORDER BY created_at DESC "
                    "LIMIT ?",
                    (post_id, time_top_n_comments,),
                )
            comments_results = self.db_cursor.fetchall()

            # Convert each comment's result into dictionary format
            comments = [{
                "comment_id":
                    comment_id,
                "post_id":
                    post_id,
                "user_id":
                    user_id,
                "content":
                    content,
                "created_at":
                    created_at,
                **({
                    "score": num_likes - num_dislikes
                } if self.show_score else {
                       "num_likes": num_likes,
                       "num_dislikes": num_dislikes
                   }),
            } for (
                comment_id,
                post_id,
                user_id,
                content,
                created_at,
                num_likes,
                num_dislikes,
            ) in comments_results]

            # Add warning message if the post has been reported
            if num_reports >= self.report_threshold:
                if is_zh_locale(self.locale):
                    warning_message = f"[警告：这条帖子已被举报 {num_reports} 次]"
                else:
                    warning_message = ("[Warning: This post has been reported"
                                       f" {num_reports} times]")
                post_content = f"{warning_message}\n{post_content}"

            # Add post information and corresponding comments to the posts list
            posts.append({
                "post_id":
                    post_id
                    if post_type_result["type"] != "repost" else original_post_id,
                "user_id":
                    user_id,
                "content":
                    post_content,
                "type":
                    post_type_result["type"],
                "created_at":
                    created_at,
                **({
                    "score": num_likes - num_dislikes
                } if self.show_score else {
                       "num_likes": num_likes,
                       "num_dislikes": num_dislikes
                   }),
                "num_shares":
                    num_shares,
                "num_reports":
                    num_reports,
                "comments":
                    comments,
                "post_attr":
                    post_attr,
                "is_following":
                    post_id in following_posts_ids,
            })
        posts_with_user_names = self._add_user_names(posts)
        return posts_with_user_names

    def _get_interaction(self, agent_id, time_top_n_comments):

        original_user_id = agent_id
        original_user_name = self._get_user_names([original_user_id,])[0]
        interaction_query = (
            "SELECT interaction_id, user_id, post_id, comment_id, original_user_id, is_mention, created_at "
            "FROM interaction WHERE original_user_id = ?")
        self.db_cursor.execute(interaction_query, (original_user_id,))
        rows = self.db_cursor.fetchall()
        notifications = []
        related_msg = None
        for row in rows:
            interaction_id, user_id, post_id, comment_id, original_user_id, is_mention, created_at = row
            user_name = self._get_user_names([user_id,])[0]
            num_likes = 0
            num_shares = 0
            if is_mention == 1:
                if comment_id is None:
                    self.db_cursor.execute(
                        "SELECT content, original_post_id, quote_content, num_likes, num_shares FROM post WHERE post_id = ?",
                        (post_id,)
                    )
                    (content, original_post_id, quote_content, num_likes, num_shares) = self.db_cursor.fetchone()
                    if original_post_id:
                        msg = f"@{user_name} mentioned @{original_user_name} in a post. Post content: {quote_content}"
                    else:
                        msg = f"@{user_name} mentioned @{original_user_name} in a post. Post content: {content}"
                else:
                    self.db_cursor.execute(
                        "SELECT content,num_likes FROM comment WHERE comment_id = ?",
                        (comment_id,)
                    )
                    (content,num_likes) = self.db_cursor.fetchone()
                    msg = f"@{user_name} mentioned @{original_user_name} in a comment. Comment content: {content}"
            else:
                if comment_id is None:
                    self.db_cursor.execute(
                        "SELECT content, quote_content, num_likes, num_shares FROM post WHERE post_id = ?",
                        (post_id,)
                    )
                    (content, quote_content, num_likes, num_shares) = self.db_cursor.fetchone()
                    related_msg = f"@{original_user_name} posted: {content}"
                    msg = f"@{user_name} quoted @{original_user_name}'s previous post. Quote content: {quote_content}"
                else:
                    self.db_cursor.execute(
                        "SELECT post_id, content,num_likes FROM comment WHERE comment_id = ?",
                        (comment_id,)
                    )
                    (post_id,content,num_likes) = self.db_cursor.fetchone()
                    self.db_cursor.execute(
                        "SELECT content FROM post WHERE post_id = ?",
                        (post_id,)
                    )
                    (post_content) = self.db_cursor.fetchone()
                    related_msg = f"@{original_user_name} posted: {post_content}"
                    msg = f"@{user_name} commented @{original_user_name}'s previous post. comment content:{content}"

            # For each post, query its corresponding comments,
            #改为None之后，再测试一遍
            comments = None
            if comment_id is None:
                try:
                    if time_top_n_comments is None:
                        self.db_cursor.execute(
                            "SELECT comment_id, post_id, user_id, content, created_at, "
                            "num_likes, num_dislikes FROM comment WHERE post_id = ?",
                            (post_id,),
                        )
                    else:
                        self.db_cursor.execute(
                            "SELECT comment_id, post_id, user_id, content, created_at, "
                            "num_likes, num_dislikes FROM comment WHERE post_id = ? "
                            "ORDER BY created_at DESC "
                            "LIMIT ?",
                            (post_id, time_top_n_comments,),
                        )
                    comments_results = self.db_cursor.fetchall()
                    # Convert each comment's result into dictionary format
                    comments = [{
                        "comment_id":
                            comment_id,
                        "post_id":
                            post_id,
                        "user_id":
                            user_id,
                        "content":
                            content,
                        "created_at":
                            created_at,
                        **({
                               "score": num_likes - num_dislikes
                           } if self.show_score else {
                            "num_likes": num_likes,
                            "num_dislikes": num_dislikes
                        }),
                    } for (
                        comment_id,
                        post_id,
                        user_id,
                        content,
                        created_at,
                        num_likes,
                        num_dislikes,
                    ) in comments_results]
                except Exception as e:
                    print(f"Error fetching comments for post_id {post_id}: {e}")
                    comments = []

            notifications.append({
                "user_id": user_id,
                "post_id": post_id,
                "comment_id": comment_id,
                "is_mention": is_mention,
                "content": msg,
                "related_content": related_msg,
                "comments": comments,
                "created_at": created_at,
                "num_likes": num_likes,
                "num_shares": num_shares,
            })
        notifications_with_user_name = self._add_user_names(notifications)
        interaction_ids = [row[0] for row in rows]
        if interaction_ids:
            placeholders = ",".join("?" * len(interaction_ids))
            delete_sql = f"DELETE FROM interaction WHERE interaction_id IN ({placeholders})"
            self.db_cursor.execute(delete_sql, interaction_ids)


        return notifications_with_user_name

    def _record_trace(self,
                      user_id,
                      action_type,
                      action_info,
                      current_time):
        r"""If, in addition to the trace, the operation function also records
        time in other tables of the database, use the time of entering
        the operation function for consistency.

        Pass in current_time to make, for example, the created_at in the post
        table exactly the same as the time in the trace table.

        If only the trace table needs to record time, use the entry time into
        _record_trace as the time for the trace record.
        """
        # if self.recsys_type == RecsysType.REDDIT:
        #     current_time = self.sandbox_clock.time_transfer(
        #         datetime.now(), self.start_time)
        # else:
        #     current_time = self.sandbox_clock.get_time_step()

        trace_insert_query = (
            "INSERT INTO trace (user_id, created_at, action, info) "
            "VALUES (?, ?, ?, ?)")
        action_info_str = json.dumps(action_info)
        self._execute_db_command(
            trace_insert_query,
            (user_id, current_time, action_type, action_info_str),
            commit=True,
        )

    def _record_interaction(self,user_id, post_id, comment_id, root_id, is_mention, interaction_type):
        r"""Record interaction such as mention, quote, comment into interaction table.
        """
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()

        if interaction_type in [ActionType.MENTION.value, ActionType.FOLLOW.value]:
            original_user_id = root_id #对于直接和人相关的动作，直接传递用户id
        else:
            original_user_id_query = "SELECT user_id FROM post WHERE post_id = ?"
            original_post_id = root_id if root_id is not None else post_id
            self._execute_db_command(original_user_id_query, (original_post_id,))
            original_user_id = self.db_cursor.fetchone()[0]

        interaction_insert_query = (
            "INSERT INTO interaction (user_id, post_id, comment_id, original_user_id, is_mention, interaction_type, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)")
        self._execute_db_command(
            interaction_insert_query,
            (user_id, post_id, comment_id, original_user_id, is_mention, interaction_type, current_time),
            commit=True,
        )



    def _check_self_post_rating(self, post_id, user_id):
        self_like_check_query = "SELECT user_id FROM post WHERE post_id = ?"
        self._execute_db_command(self_like_check_query, (post_id,))
        result = self.db_cursor.fetchone()
        if result and result[0] == user_id:
            error_message = ("Users are not allowed to like/dislike their own "
                             "posts.")
            return {"success": False, "error": error_message}
        else:
            return None


    def _check_self_comment_rating(self, comment_id, user_id):
        self_like_check_query = ("SELECT user_id FROM comment WHERE "
                                 "comment_id = ?")
        self._execute_db_command(self_like_check_query, (comment_id,))
        result = self.db_cursor.fetchone()
        if result and result[0] == user_id:
            error_message = ("Users are not allowed to like/dislike their "
                             "own comments.")
            return {"success": False, "error": error_message}
        else:
            return None


    def _get_post_type(self, post_id: int):
        query = (
            "SELECT original_post_id, quote_content FROM post WHERE post_id "
            "= ?")
        self._execute_db_command(query, (post_id,))
        result = self.db_cursor.fetchone()

        if not result:
            return None

        original_post_id, quote_content = result

        if original_post_id is None:
            # common post without quote or repost
            return {"type": "common", "root_post_id": None}
        elif quote_content is None:
            # post with repost
            return {"type": "repost", "root_post_id": original_post_id}
        else:
            # post with quote
            return {"type": "quote", "root_post_id": original_post_id}


    def _get_user_names(self, agent_ids):
        # Get all unique agent IDs to avoid duplicate queries
        unique_agent_ids = list(set(agent_ids))

        # Query database for unique IDs only
        placeholders = ",".join("?" * len(unique_agent_ids))
        user_id_query = f"SELECT agent_id, name FROM user WHERE agent_id IN ({placeholders})"
        self._execute_db_command(user_id_query, unique_agent_ids)
        results = self.db_cursor.fetchall()

        # Create mapping from agent_id to name
        id_to_name = {result[0]: result[1] for result in results}

        # Return names in the same order as the input agent_ids, preserving duplicates
        return [id_to_name.get(agent_id) for agent_id in agent_ids]

    def _add_user_names(self, posts):
        user_ids = [post["user_id"] for post in posts]

        user_names = self._get_user_names(user_ids)

        for post, user_name in zip(posts, user_names):
            post["user_name"] = user_name
            if post["comments"]:
                comments_user_ids = [comment["user_id"] for comment in  post["comments"]]
                comments_user_names = self._get_user_names(comments_user_ids)
                for comment, comments_user_name in zip(post["comments"], comments_user_names):
                    comment["user_name"] = comments_user_name

        return posts
