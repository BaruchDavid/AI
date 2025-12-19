from config.diagnostic_config import DiagnosticConfig
import yaml


def load_diagnostic_config(path: str) -> DiagnosticConfig:
    with open(path, "r") as config_file:
        data = yaml.safe_load(config_file)
    return DiagnosticConfig(**data["diagnostics"])
