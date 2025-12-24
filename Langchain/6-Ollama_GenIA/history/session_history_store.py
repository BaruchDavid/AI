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
        self.logger.debug(f"SessionHistoryStore created, id={id(self)}")

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        self.logger.debug(f"prüfe die session {session_id} im session_store")
        if session_id not in self._store:
            self.logger.debug(f"session_id {session_id} ist nicht im session_store")
            self._store[session_id] = InMemoryChatHistory()
        else:
            self.logger.debug(f"session_id {session_id} ist im session_store")
        history = self._store[session_id]
        self.logger.debug(f"session history: {history}")
        return self._store[session_id]

    def clear_history(self, session_id: str) -> None:
        if session_id in self._store:
            self._store[session_id].clear()

    def clear_all(self) -> None:
        self._store.clear()
