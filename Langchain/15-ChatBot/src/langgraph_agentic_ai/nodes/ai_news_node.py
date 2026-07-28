from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate
import os


class AINewsNode:
    def __init__(self, llm):
        self.llm = llm
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))  # Replace with your
        self.state = {}

    def fetch_news(self, state: dict) -> dict:
        """
        Fetches news articles based on the provided query using the Tavily API.

        Args:
            query (str): The search query for fetching news articles.

        Returns:
            list: A list of news articles.
        """
        frequency = state["messages"][0].content.lower()
        self.state["frequency"] = frequency
        time_range_map = {"last 24 hours": "d", "last 7 days": "w", "last 30 days": "m"}
        days_map = {"last 24 hours": 1, "last 7 days": 7, "last 30 days": 30}

        response = self.tavily_client.search(
            query="Top Artificial Intelligence (AI technology news Austria and globally)",
            topic="news",
            time_range=time_range_map[frequency],
            include_answer="advanced",
            max_results=2,
            days=days_map[frequency],
        )

        state["news_data"] = response.get("results", [])
        self.state["news_data"] = state["news_data"]
        return state

    def summarize_news(self, state: dict) -> dict:
        """
        Summarizes the fetched news articles using the provided LLM.

        Args:
            state (dict): The current state containing the fetched news articles.

        Returns:
            dict: The updated state with the summarized news.
        """
        news_items = state.get("news_data", [])
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Summarize AI news articles into markdown format. For each item include:
                - Title
                - Date in **DD.MM.YYYY** format in IST timezone
                - Concise sentences summary from latest news
                - Sort news by date wise (latest first)
                - Source URL as link
                Use format:
                ### [Date]
                - [Summary](URL)""",
                ),
                ("user", "Please summarize the following news articles:\n{news_content}"),
            ]
        )

        articles_str = "\n\n".join(
            [f"Content: {item.get('content', '')}\nURL: {item.get('url', '')}\nDate: {item.get('published_date', '')}" for item in news_items]
        )

        response = self.llm.invoke(prompt_template.format(news_content=articles_str))
        state["summary"] = response.content
        self.state["summary"] = state["summary"]
        return self.state

    def save_result(self, state):
        """
        Saves the summarized news to a file.

        Args:
            state (dict): The current state containing the summarized news.

        Returns:
            dict: The updated state after saving the summary.
        """
        frequency = self.state["frequency"]
        summary = state["summary"]
        filename = f"./AINews/{frequency}_summary.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"### AI News Summary ({frequency.capitalize()})\n\n")
            f.write(summary)
        self.state["filename"] = filename
        return self.state
