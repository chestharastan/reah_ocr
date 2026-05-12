import importlib
from pathlib import Path

# Auto-discover every .py file in this directory and one level of subdirectories.
# Each file must export:  Model (nn.Module subclass)  +  get_transform(h, w)
# The filename (without .py) becomes the architecture name in config.yml.

_here = Path(__file__).parent
REGISTRY = {}

for _path in sorted(_here.glob("*.py")):
    if _path.stem == "__init__":
        continue
    _module = importlib.import_module(f"architectures.{_path.stem}")
    REGISTRY[_path.stem] = (_module.Model, _module.get_transform)

for _subdir in sorted(_here.iterdir()):
    if not _subdir.is_dir() or _subdir.name.startswith("_"):
        continue
    for _path in sorted(_subdir.glob("*.py")):
        if _path.stem == "__init__":
            continue
        _module = importlib.import_module(f"architectures.{_subdir.name}.{_path.stem}")
        REGISTRY[_path.stem] = (_module.Model, _module.get_transform)


def build_model(config, num_classes):
    name = config["model"]["architecture"]
    if name not in REGISTRY:
        raise ValueError(f"Unknown architecture '{name}'. Available: {list(REGISTRY)}")
    cls, _ = REGISTRY[name]
    return cls(
        num_classes=num_classes,
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )


def build_transform(config):
    name = config["model"]["architecture"]
    if name not in REGISTRY:
        raise ValueError(f"Unknown architecture '{name}'. Available: {list(REGISTRY)}")
    _, transform_fn = REGISTRY[name]
    return transform_fn(
        config["preprocessing"]["image_height"],
        config["preprocessing"]["image_width"],
    )
