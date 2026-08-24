import yaml
from pathlib import Path
from types import SimpleNamespace

def _dict_to_sns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_sns(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_sns(i) for i in d]
    return d

def load_config(path="config.yaml"):
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw["paths"] = {k: Path(v).resolve() for k, v in raw["paths"].items()}
    return _dict_to_sns(raw)

if __name__ == "__main__":
    cfg = load_config()
    print("Config successfully loaded!")
