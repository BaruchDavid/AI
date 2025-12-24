import logging
from langchain_core.chat_history import BaseChatMessageHistory
from history.in_memory_chat_history import InMemoryChatHistory


"""
Verwaltung vieler Konversationen
Session-ID → ChatHistory"""


class SessionHistoryStore:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._store: dict[str, BaseChatMessageHistory] = {}

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._store:
            self.logger.info(
                f"no messages for this session_id {session_id}, create new history"
            )
            self._store[session_id] = InMemoryChatHistory()
        return self._store[session_id]

    def clear_history(self, session_id: str) -> None:
        self.logger.info(f"clean history for session_id {session_id}")
        if session_id in self._store:
            self._store[session_id].clear()

    def clear_all(self) -> None:
        self.logger.info("clean whole session_store")
        self._store.clear()
