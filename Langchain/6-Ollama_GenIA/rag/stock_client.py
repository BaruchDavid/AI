import requests
import logging


class StockClient:

    _alpha_avantage_url = "https://www.alphavantage.co/query" 
    _params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": "SPY",
        "apikey": "",
    }

    def __init__(self, *, api_key):
        self._params("apikey") = api_key
        self._logger = logging.getLogger(__name__)

    def check_stocks(self) -> str:
        try:
            response= requests.get(self._alpha_avantage_url, params=self._params, timeout=10,)
            response.raise_for_status()
            return response.json()
            
        except Exception:
           self.logger.exception("Retrieving new Stock-Information")
           raise

        

        
