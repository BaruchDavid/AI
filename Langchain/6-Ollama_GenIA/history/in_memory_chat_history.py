import logging
from langchain_core.chat_history import BaseChatMessageHistory
from typing_extensions import override

""" 
eine einzelne Konversation
Nachrichtenliste
gehört genau EINER Session """


class InMemoryChatHistory(BaseChatMessageHistory):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.messages = []

    def add_messages(self, messages):
        self.logger.debug(f"Added message: {messages}")
        self.messages.extend(messages)
        self.logger.debug(f"all messages at now: {self.messages}")

    @override
    def clear(self):
        self.messages = []
