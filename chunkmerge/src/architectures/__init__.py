import importlib
from pathlib import Path

# Auto-discover every .py file in this directory and one level of subdirectories.
# Each module must export:
#   Model: nn.Module subclass with signature (num_classes, pad_id, bos_id, eos_id, **cfg_kwargs)
#   get_transform(image_height, image_width)

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


def build_model(config, num_classes, pad_id, bos_id, eos_id):
    name = config["model"]["architecture"]
    if name not in REGISTRY:
        raise ValueError(f"Unknown architecture '{name}'. Available: {list(REGISTRY)}")
    cls, _ = REGISTRY[name]

    # Pass every model.* config key (except the architecture name itself) to
    # the model constructor. Lets each architecture file declare its own
    # hyperparameters in the config without having to update build_model.
    kwargs = {k: v for k, v in config["model"].items() if k != "architecture"}
    return cls(
        num_classes=num_classes,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        **kwargs,
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
