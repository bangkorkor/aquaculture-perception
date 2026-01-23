import argparse
from ultralytics import YOLO

from utils.io import read_json
from utils.paths import objdet_root, resolve_from_objdet
from utils.run_registry import load_runs_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="Run id in runs.csv")
    ap.add_argument("--dry", action="store_true", help="Print resolved config and exit")
    args = ap.parse_args()

    root = objdet_root()

    runs = load_runs_csv(root / "runs.csv")
    if args.id not in runs:
        raise KeyError(f"Run '{args.id}' not found. Available: {', '.join(sorted(runs.keys()))}")

    run = runs[args.id]

    param_sets = read_json(root / "configs" / "training-params.json").get("sets", {})
    params_id = (run.get("params_id") or "default").strip()
    if params_id not in param_sets:
        raise KeyError(f"params_id '{params_id}' not found. Available: {', '.join(sorted(param_sets.keys()))}")

    # All training kwargs come from training-params.json
    train_kwargs = dict(param_sets[params_id])

    # Required per-run fields
    data_yaml = resolve_from_objdet(run["data"])
    model_cfg = resolve_from_objdet(run["model"])

    train_kwargs["data"] = str(data_yaml)

    # Optional per-run routing fields
    if run.get("project"):
        train_kwargs["project"] = run["project"]
    if run.get("name"):
        train_kwargs["name"] = run["name"]

    # Optional pretrained checkpoint
    pretrained = resolve_from_objdet(run["pretrained"]) if run.get("pretrained") else None

    if args.dry:
        print({
            "id": args.id,
            "model": str(model_cfg),
            "pretrained": str(pretrained) if pretrained else None,
            "params_id": params_id,
            "train_kwargs": train_kwargs
        })
        return

    model = YOLO(str(model_cfg),  task="detect")
    if pretrained:
        model = model.load(str(pretrained))

    model.info(verbose=True)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
