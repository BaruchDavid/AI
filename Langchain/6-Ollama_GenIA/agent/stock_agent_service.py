import logging


class StockAgentService:
    def __init__(self, agent):
        self._agent = agent
        self._chat_history = []
        self._cache = {}
        self._logger = logging.getLogger(__name__)

    def ask(self, user_input: str) -> str:
        if user_input in self._cache:
            return self._cache[user_input]

        response = self._agent.invoke(
            {
                "input": user_input,
                "chat_history": self._chat_history,
            }
        )

        self._chat_history.append(("user", user_input))
        self._chat_history.append(("agent", response["output"]))

        self._cache[user_input] = response["output"]
        return response["output"]

    def shutdown(self):
        self._cache.clear()
        self._chat_history.clear()
