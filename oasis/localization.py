from __future__ import annotations


ZH_CN = "zh_cn"


def is_zh_locale(locale: str | None) -> bool:
    return str(locale or "").lower().replace("-", "_") == ZH_CN


def platform_display_name(platform_name: str | None, locale: str | None) -> str:
    if platform_name:
        return platform_name
    return "微博" if is_zh_locale(locale) else "Twitter"


ACTION_LABELS_ZH = {
    "sign_up": "注册用户",
    "exit": "退出平台",
    "create_post": "发布帖子",
    "like_post": "点赞帖子",
    "unlike_post": "取消点赞帖子",
    "dislike_post": "点踩帖子",
    "undo_dislike_post": "取消点踩帖子",
    "repost": "转发帖子",
    "quote_post": "引用帖子",
    "create_comment": "发表评论",
    "like_comment": "点赞评论",
    "unlike_comment": "取消点赞评论",
    "dislike_comment": "点踩评论",
    "undo_dislike_comment": "取消点踩评论",
    "follow": "关注用户",
    "unfollow": "取消关注用户",
    "mute": "屏蔽用户",
    "unmute": "取消屏蔽用户",
    "search_posts": "搜索帖子",
    "search_user": "搜索用户",
    "trend": "查看热榜",
    "refresh": "刷新信息流",
    "do_nothing": "不采取行动",
    "purchase_product": "购买商品",
    "mention": "提及用户",
    "update_rec_table": "更新推荐表",
    "interview": "接受访谈",
    "report_post": "举报帖子",
    "join_group": "加入群组",
    "leave_group": "退出群组",
    "send_to_group": "发送群消息",
    "create_group": "创建群组",
    "listen_from_group": "读取群消息",
    "view_comment": "查看评论",
    "view_comments": "查看评论",
}


TOOL_DESCRIPTIONS_ZH = {
    "sign_up": "注册一个新用户，并创建其用户名、展示名称和个人简介。",
    "refresh": "刷新并获取推荐帖子。",
    "do_nothing": "看到某条当前可见帖子后选择不互动。必须提供触发该决定的帖子 ID，用于行为一致性检查；平台执行时会忽略该参数，只记录用户未采取行动。",
    "create_post": "发布一条新帖子。内容应符合当前人物设定和平台语境；如果需要，可以提及其他用户。",
    "like_post": "点赞一条当前可见的帖子。",
    "unlike_post": "取消之前对某条帖子的点赞。",
    "dislike_post": "点踩一条帖子。",
    "undo_dislike_post": "取消之前对某条帖子的点踩。",
    "repost": "转发一条当前可见的帖子，不添加新的评论内容。",
    "quote_post": "引用一条当前可见的帖子，并附上自己的评论或反应。",
    "search_posts": "根据关键词搜索帖子。",
    "search_user": "根据关键词搜索用户。",
    "trend": "查看一段时间内的热门帖子。",
    "create_comment": "在指定帖子下发表评论。评论内容应符合人物设定和当前上下文。",
    "like_comment": "点赞指定评论。",
    "dislike_comment": "点踩指定评论。",
    "unlike_comment": "取消之前对指定评论的点赞。",
    "undo_dislike_comment": "取消之前对指定评论的点踩。",
    "follow": "因看到某条当前可见帖子而关注一名用户。必须提供触发关注行为的帖子 ID，用于行为一致性检查；平台执行时会忽略该帖子 ID，只使用要关注的用户 ID。",
    "unfollow": "取消关注一名用户。",
    "mute": "屏蔽一名用户。",
    "unmute": "取消屏蔽一名用户。",
    "purchase_product": "购买一个商品。",
    "update_rec_table": "更新平台推荐表。",
    "mention": "在帖子、引用或评论中提及其他用户。",
    "interview": "回答一条访谈问题。",
    "report_post": "举报一条帖子，并说明举报理由。",
    "join_group": "加入指定群组。",
    "leave_group": "退出指定群组。",
    "send_to_group": "向指定群组发送消息。",
    "create_group": "创建一个新群组。",
    "listen_from_group": "读取当前用户所在群组中的消息。",
    "view_comment": "查看指定帖子下的评论。",
    "view_comments": "查看指定帖子下的评论。",
}


PARAMETER_TITLES_ZH = {
    "user_name": "用户名",
    "name": "展示名称",
    "bio": "个人简介",
    "post_id": "帖子 ID",
    "content": "内容",
    "mentioned_user_ids": "提及用户 ID 列表",
    "query": "搜索关键词",
    "followee_id": "目标用户 ID",
    "mutee_id": "屏蔽用户 ID",
    "comment_id": "评论 ID",
    "product_name": "商品名称",
    "purchase_num": "购买数量",
    "reason": "原因",
    "prompt": "访谈问题",
    "report_reason": "举报原因",
    "group_name": "群组名称",
    "group_id": "群组 ID",
    "message": "消息内容",
    "messages": "文本内容",
    "type": "内容类型",
}


PARAMETER_DESCRIPTIONS_ZH = {
    "user_name": "新用户的唯一用户名。",
    "name": "新用户的展示名称。",
    "bio": "新用户的个人简介。",
    "post_id": "目标帖子的 ID。必须来自当前可见帖子或平台返回结果。",
    "content": "要发布的中文内容。应自然、口语化，并符合人物设定。",
    "mentioned_user_ids": "需要提及的用户 ID 列表；没有需要提及的用户时使用空列表。",
    "query": "搜索关键词。",
    "followee_id": "要关注或取消关注的用户 ID。",
    "mutee_id": "要屏蔽或取消屏蔽的用户 ID。",
    "comment_id": "目标评论的 ID。",
    "product_name": "要购买的商品名称。",
    "purchase_num": "要购买的商品数量。",
    "reason": "原因。",
    "prompt": "访谈问题。",
    "report_reason": "举报原因。",
    "group_name": "要创建的群组名称。",
    "group_id": "目标群组的 ID。",
    "message": "要发送到群组的消息内容。",
    "messages": "文本内容或内容列表。",
    "type": "内容类型。",
}


TOOL_PARAMETER_DESCRIPTIONS_ZH = {
    "sign_up": {
        "user_name": "用于注册的新用户名。",
        "name": "新用户在平台上显示的名称。",
        "bio": "新用户的个人简介或自我描述。",
    },
    "do_nothing": {
        "post_id": "必填。触发“不采取行动”决定的当前可见帖子 ID。请从【帖子】中选择。",
    },
    "create_post": {
        "content": "要发布的新帖子内容。应自然、口语化，符合人物设定和平台语境。",
        "mentioned_user_ids": "这条帖子需要提及的用户 ID 列表；没有需要提及时使用空列表。",
    },
    "repost": {
        "post_id": "要转发的当前可见帖子 ID。",
    },
    "quote_post": {
        "post_id": "要引用的当前可见帖子 ID。",
        "content": "引用帖子时附加的评论或反应内容，应围绕原帖表达。",
        "mentioned_user_ids": "引用内容中需要提及的用户 ID 列表；没有需要提及时使用空列表。",
    },
    "like_post": {
        "post_id": "要点赞的当前可见帖子 ID。",
    },
    "unlike_post": {
        "post_id": "要取消点赞的帖子 ID。",
    },
    "dislike_post": {
        "post_id": "要点踩的当前可见帖子 ID。",
    },
    "undo_dislike_post": {
        "post_id": "要取消点踩的帖子 ID。",
    },
    "search_posts": {
        "query": "用于搜索帖子的关键词。",
    },
    "search_user": {
        "query": "用于搜索用户的关键词，可以是用户名、昵称或简介中的信息。",
    },
    "follow": {
        "followee_id": "要关注的用户 ID。优先选择触发关注行为的当前可见帖子的作者 ID。",
        "post_id": "必填。触发这次关注行为的当前可见帖子 ID。请从【帖子】中选择；平台执行关注时不会实际使用该参数。",
    },
    "unfollow": {
        "followee_id": "要取消关注的用户 ID。",
    },
    "mute": {
        "mutee_id": "要屏蔽的用户 ID。",
    },
    "unmute": {
        "mutee_id": "要取消屏蔽的用户 ID。",
    },
    "create_comment": {
        "post_id": "要发表评论的目标帖子 ID。",
        "content": "要发表的评论内容。应自然、口语化，并结合目标帖子。",
        "mentioned_user_ids": "评论中需要提及的用户 ID 列表；没有需要提及时使用空列表。",
    },
    "like_comment": {
        "comment_id": "要点赞的评论 ID。",
    },
    "unlike_comment": {
        "comment_id": "要取消点赞的评论 ID。",
    },
    "dislike_comment": {
        "comment_id": "要点踩的评论 ID。",
    },
    "undo_dislike_comment": {
        "comment_id": "要取消点踩的评论 ID。",
    },
    "purchase_product": {
        "product_name": "要购买的商品名称。",
        "purchase_num": "要购买的商品数量。",
    },
    "interview": {
        "prompt": "需要当前用户回答的访谈问题。",
    },
    "report_post": {
        "post_id": "要举报的帖子 ID。",
        "report_reason": "举报该帖子的原因。",
    },
    "create_group": {
        "group_name": "要创建的新群组名称。",
    },
    "join_group": {
        "group_id": "要加入的群组 ID。",
    },
    "leave_group": {
        "group_id": "要退出的群组 ID。",
    },
    "send_to_group": {
        "group_id": "要发送消息的目标群组 ID。",
        "message": "要发送到群组中的消息内容。",
    },
    "view_comment": {
        "post_id": "要查看评论的帖子 ID。",
    },
}


ERROR_MESSAGES_ZH = {
    "No such product.": "未找到该商品。",
    "Content cannot be empty.": "内容不能为空。",
    "No posts found.": "没有找到帖子。",
    "Fail to get latest posts count": "获取最新帖子数量失败。",
    "Repost record already exists.": "已经转发过这条帖子。",
    "Post not found.": "未找到该帖子。",
    "Like record already exists.": "已经点赞过这条帖子。",
    "Like record does not exist.": "尚未点赞过这条帖子。",
    "Dislike record already exists.": "已经点踩过这条帖子。",
    "Dislike record does not exist.": "尚未点踩过这条帖子。",
    "No posts found matching the query.": "没有找到匹配搜索条件的帖子。",
    "No users found matching the query.": "没有找到匹配搜索条件的用户。",
    "Follow record already exists.": "已经关注过该用户。",
    "Follow record does not exist.": "尚未关注该用户。",
    "Mute record already exists.": "已经屏蔽过该用户。",
    "No mute record exists.": "尚未屏蔽该用户。",
    "No trending posts in the specified period.": "指定时间范围内没有热门帖子。",
    "Comment like record already exists.": "已经点赞过这条评论。",
    "Comment like record does not exist.": "尚未点赞过这条评论。",
    "Comment dislike record already exists.": "已经点踩过这条评论。",
    "Comment dislike record does not exist.": "尚未点踩过这条评论。",
    "Report record already exists.": "已经举报过这条帖子。",
    "User is not a member of this group.": "用户不是该群组成员。",
    "Group does not exist.": "群组不存在。",
    "User is already in the group.": "用户已经在该群组中。",
    "No comments found for the post.": "该帖子下没有找到评论。",
}


def zh_action_label(action_name: str) -> str:
    return ACTION_LABELS_ZH.get(action_name, action_name)


def localize_result_text(result: dict, locale: str | None) -> dict:
    if not is_zh_locale(locale) or not isinstance(result, dict):
        return result
    localized = dict(result)
    for key in ("error", "message"):
        value = localized.get(key)
        if not isinstance(value, str):
            continue
        if value.startswith("Content exceeds character limit."):
            localized[key] = value.replace(
                "Content exceeds character limit. Length:",
                "内容超过字数限制。当前长度：",
            ).replace(", Limit:", "，上限：")
        elif value.startswith("Agent ") and "has not signed up" in value:
            localized[key] = "该智能体尚未注册，当前没有对应的平台用户 ID。"
        elif value.startswith("Users are not allowed to like/dislike their own posts"):
            localized[key] = "用户不允许点赞或点踩自己的帖子。"
        elif value.startswith("Users are not allowed to like/dislike their own comments"):
            localized[key] = "用户不允许点赞或点踩自己的评论。"
        elif value.startswith("Post not found. post_id:"):
            localized[key] = value.replace(
                "Post not found. post_id:",
                "未找到帖子，post_id：")
        else:
            localized[key] = ERROR_MESSAGES_ZH.get(value, value)
    return localized


def localize_tool_schema(tool_schema: dict, locale: str | None) -> dict:
    if not is_zh_locale(locale):
        return tool_schema
    function = tool_schema.get("function", {})
    name = function.get("name")
    if name in TOOL_DESCRIPTIONS_ZH:
        function["description"] = TOOL_DESCRIPTIONS_ZH[name]
    properties = function.get("parameters", {}).get("properties", {})
    for param_name, param_schema in properties.items():
        if param_name in PARAMETER_TITLES_ZH:
            param_schema["title"] = PARAMETER_TITLES_ZH[param_name]
        action_param_descriptions = TOOL_PARAMETER_DESCRIPTIONS_ZH.get(
            name,
            {},
        )
        description = action_param_descriptions.get(
            param_name,
            PARAMETER_DESCRIPTIONS_ZH.get(param_name),
        )
        if description:
            param_schema["description"] = description
    return tool_schema
