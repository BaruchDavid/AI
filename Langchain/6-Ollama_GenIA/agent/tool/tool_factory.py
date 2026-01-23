from langchain.tools import tool
from rag.client.stock_client import StockClient
import json


def build_stock_tool(stock_client: StockClient):

    def fetch_spy_stock() -> str:
        """
        Holt tägliche SPY-Aktienkurse von Alpha Vantage.
        Nutze dieses Tool, wenn aktuelle Marktdaten benötigt werden.
        """
        return json.dumps(stock_client.check_stocks())
