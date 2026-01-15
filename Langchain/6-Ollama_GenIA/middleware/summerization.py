from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, SystemMessage


class Summerization:
    """
    model: das Modell, das zum Erstellen der Zusammenfassungen verwendet wird
    trigger: Bedingungen, wann es triggern soll (z. B. ab X Tokens)
    keep: wieviel Kontext nach einer Zusammenfassung erhalten bleiben soll
    """

    def get_message_summerization(
        self, *, model_name, summerizaion_limit
    ) -> SummarizationMiddleware:
        return SummarizationMiddleware(
            model=model_name,
            trigger=("messages", summerizaion_limit),
            keep=("messages", 4),
        )

    def get_token_summerization(
        self, *, model_name, token_limit
    ) -> SummarizationMiddleware:
        return SummarizationMiddleware(
            model=model_name,
            trigger=("tokens", token_limit),
            keep=("tokens", 200),
        )

    def get_fraction_summerization(
        self, *, model_name, fraction_limit
    ) -> SummarizationMiddleware:
        return SummarizationMiddleware(
            model_name, trigger=("fraction", fraction_limit), keep=("fraction", 0.002)
        )

    def execute_agent(self, *, llm, selected_middelware):
        agent = create_agent(
            model=llm,
            tools=[],
            checkpointer=InMemorySaver,
            middleware=selected_middelware,
        )
