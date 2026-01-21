import json
from pathlib import Path
import logging


class StockManager:
    def __init__(self, *, data: str):
        self._content = data
        self._logger = logging.getLogger(__name__)

    def write_data_file(self, *, data):

        if "Information" in data:
            logging.info("Premium-Endpunkt, nicht verfügbar für Free-Key")
        elif "Note" in data:
            logging.info("API-Call Limit erreicht")

        time_series = data.get("Time Series (Daily)")
        if not time_series:
            raise ValueError("Keine Kursdaten gefunden")

        base_path = Path(__file__).resolve().parent
        json_path = base_path / "spy_data.json"

        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(time_series, f, ensure_ascii=False, indent=4)

        self._logger.info(f"Kursdaten erfolgreich gespeichert: {path}")
