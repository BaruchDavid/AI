import requests
import logging
from typing import Any
import json
from pathlib import Path


class StockClient:

    _alpha_avantage_url = "https://www.alphavantage.co/query"

    def __init__(self, *, api_key, http=None):
        self.__http = http or requests
        self._logger = logging.getLogger(__name__)
        self._params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
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
            if "Information" in data:
                print("Premium-Endpunkt, nicht verfügbar für Free-Key")
            elif "Note" in data:
                print("API-Call Limit erreicht")
            else:
                # hier sind die echten Kursdaten
                time_series = data.get("Time Series (Daily)")
                for date, daily in time_series.items():
                    print(date, daily["1. open"], daily["4. close"])

            # Die echten Kursdaten extrahieren
            time_series = data.get("Time Series (Daily)")
            if not time_series:
                raise ValueError("Keine Kursdaten gefunden")

            # Absoluter Pfad der aktuell ausgeführten Datei
            base_path = Path(__file__).resolve().parent
            json_path = base_path / "spy_data.json"

            # JSON-Datei schreiben
            path = Path(json_path)
            path.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ordner erstellen, falls nicht vorhanden
            with open(path, "w", encoding="utf-8") as f:
                json.dump(time_series, f, ensure_ascii=False, indent=4)

            self._logger.info(f"Kursdaten erfolgreich gespeichert: {file_path}")

        except Exception:
            self.logger.exception("Retrieving new Stock-Information")
            raise
