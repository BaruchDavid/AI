from __future__ import annotations
from pathlib import Path
from typing import Optional
import yaml
import json


class PromptLoader:

    _instance: Optional[PromptLoader] = None
    _initialized: bool = False

    # __new__ für Singelton verwenden
    def __new__(cls, yaml_path: str = None):

        if cls._instance is None:
            # konvention, man kann auch object.__new__(cls) verwenden
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, yaml_path: str = None):

        if self._initialized:
            return

        if yaml_path is None:
            yaml_path = Path(__file__).parent / "prompts.yaml"

        self.yaml_path = Path(yaml_path)
        self.prompts = self._load_yaml()

        PromptLoader._initialized = True

    @classmethod
    def get_instance(cls, yaml_path: str = None) -> PromptLoader:

        if cls._instance is None:
            cls._instance = cls(yaml_path)  ## ← Erstelle Instanz der EIGENEN Klasse
        return cls._instance

    @classmethod
    def reset_instance(cls):

        cls._instance = None
        cls._initialized = False

    def _load_yaml(self) -> dict:

        with open(self.yaml_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def get_prompt(self, agent_name: str) -> str:

        if agent_name not in self.prompts:
            raise KeyError(f"Agent '{agent_name}' not found in prompts.yaml")

        prompt_data = self.prompts[agent_name]

        if isinstance(prompt_data, str):
            return prompt_data

        if isinstance(prompt_data, dict):
            return self._format_dict_to_prompt(prompt_data)

    def _format_dict_to_prompt(self, prompt_dict: dict) -> str:
        lines = []

        for key, value in prompt_dict.items():
            # Key als lesbaren Titel formatieren
            title = key.replace("_", " ").title()

            if isinstance(value, str):
                # Langer String: Direkt ohne Titel
                if len(value) > 100 or "\n" in value:
                    lines.append(value)
                else:
                    lines.append(f"{title}: {value}")

            elif isinstance(value, list):
                # Liste als natürliche Aufzählung
                lines.append(f"{title}:")
                for item in value:
                    lines.append(f"- {item}")

            elif isinstance(value, dict):
                # Rekursiv
                lines.append(f"{title}:")
                lines.append(self._format_dict_to_prompt(value))

            else:
                lines.append(f"{title}: {value}")

            lines.append("")  # Leerzeile zwischen Sections

        # Entferne überflüssige Leerzeilen am Ende
        result = "\n".join(lines)
        while "\n\n\n" in result:
            result = result.replace("\n\n\n", "\n\n")

        return result.strip()
