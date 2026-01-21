import requests
import logging
from typing import Any


class StockClient:

    _alpha_avantage_url = "https://www.alphavantage.co/query"

    def __init__(self, *, api_key, http=None):
        self.__http = http or requests
        self._logger = logging.getLogger(__name__)
        self._params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": "SPY",
            "apikey": api_key,
        }

    def check_stocks(self) -> Any:
        try:
            response = self.__http.get(
                self._alpha_avantage_url,
                params=self._params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

        except Exception:
            self.logger.exception("Retrieving new Stock-Information")
            raise
