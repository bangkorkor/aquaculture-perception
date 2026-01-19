from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

def objdet_root() -> Path:
    # .../object-detection/utils/paths.py -> object-detection/
    return Path(__file__).resolve().parents[1]

def resolve_from_objdet(path: PathLike) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (objdet_root() / p).resolve()
