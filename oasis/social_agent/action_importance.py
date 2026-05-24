from __future__ import annotations

from oasis.social_platform.typing import ActionType


_HIGH_IMPORTANCE_ACTIONS = frozenset({
    ActionType.CREATE_POST.value,
    ActionType.QUOTE_POST.value,
    ActionType.CREATE_COMMENT.value,
    ActionType.SEND_TO_GROUP.value,
})

_MEDIUM_IMPORTANCE_ACTIONS = frozenset({
    ActionType.REPOST.value,
    ActionType.FOLLOW.value,
    ActionType.REPORT_POST.value,
    ActionType.DISLIKE_POST.value,
})

_LOW_IMPORTANCE_ACTIONS = frozenset({
    ActionType.LIKE_POST.value,
    ActionType.UNLIKE_POST.value,
    ActionType.UNDO_DISLIKE_POST.value,
    ActionType.LIKE_COMMENT.value,
    ActionType.DISLIKE_COMMENT.value,
})


def heuristic_action_importance(action_name: str | ActionType | None) -> int:
    """Return a deterministic memory-importance score for an action type."""
    if isinstance(action_name, ActionType):
        action_name = action_name.value
    action_name = str(action_name or "")
    if action_name in _HIGH_IMPORTANCE_ACTIONS:
        return 7
    if action_name in _MEDIUM_IMPORTANCE_ACTIONS:
        return 5
    if action_name in _LOW_IMPORTANCE_ACTIONS:
        return 3
    return 1
