from __future__ import annotations

from typing import List, Optional, Tuple

from camel.memories.base import BaseContextCreator
from camel.memories.records import ContextRecord
from camel.messages import OpenAIMessage
from camel.types import OpenAIBackendRole
from camel.utils import BaseTokenCounter


class RecentTokenLimitContextCreator(BaseContextCreator):
    r"""Create context by keeping the newest records within a token budget.

    CAMEL's default ``ScoreBasedContextCreator`` currently counts tokens but
    does not hard-filter records by ``token_limit``. This creator preserves the
    first system/developer message and then walks backward through the
    remaining history, keeping only the newest records that fit.
    """

    def __init__(
        self,
        token_counter: BaseTokenCounter,
        token_limit: int,
    ) -> None:
        if token_limit <= 0:
            raise ValueError("`token_limit` must be a positive integer.")
        self._token_counter = token_counter
        self._token_limit = token_limit

    @property
    def token_counter(self) -> BaseTokenCounter:
        return self._token_counter

    @property
    def token_limit(self) -> int:
        return self._token_limit

    def create_context(
        self,
        records: List[ContextRecord],
    ) -> Tuple[List[OpenAIMessage], int]:
        if not records:
            return [], 0

        system_record: Optional[ContextRecord] = None
        history_records: List[ContextRecord] = []

        for record in records:
            role = record.memory_record.role_at_backend
            if (
                system_record is None
                and role in {
                    OpenAIBackendRole.SYSTEM,
                    OpenAIBackendRole.DEVELOPER,
                }
            ):
                system_record = record
            else:
                history_records.append(record)

        history_records.sort(key=lambda record: record.timestamp)

        fixed_messages: List[OpenAIMessage] = []
        if system_record is not None:
            fixed_messages.append(system_record.memory_record.to_openai_message())

        fixed_tokens = self._count_tokens(fixed_messages)
        if fixed_tokens > self.token_limit:
            return [], 0

        if not history_records:
            return fixed_messages, fixed_tokens

        selected: List[ContextRecord] = []
        total_tokens = fixed_tokens
        low = 0
        high = len(history_records)
        while low < high:
            mid = (low + high) // 2
            candidate_records = history_records[mid:]
            candidate_messages = [
                *fixed_messages,
                *[
                    item.memory_record.to_openai_message()
                    for item in candidate_records
                ],
            ]
            candidate_tokens = self._count_tokens(candidate_messages)
            if candidate_tokens <= self.token_limit:
                selected = candidate_records
                total_tokens = candidate_tokens
                high = mid
            else:
                low = mid + 1

        messages = [
            *fixed_messages,
            *[record.memory_record.to_openai_message() for record in selected],
        ]
        return messages, total_tokens

    def _count_tokens(self, messages: List[OpenAIMessage]) -> int:
        if not messages:
            return 0
        return self.token_counter.count_tokens_from_messages(messages)
