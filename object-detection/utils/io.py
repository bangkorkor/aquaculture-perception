import json
from pathlib import Path
from typing import Any, Dict, Union

PathLike = Union[str, Path]

def read_json(path: PathLike) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text())
