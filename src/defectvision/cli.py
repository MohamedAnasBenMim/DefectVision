"""Command-line interface: `dv <command>` (see README)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dv", description="DefectVision")
    p.add_argument("--config", default="configs/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-data", help="download the dataset (pinned commit)")
    sub.add_parser("fetch-weights", help="download ImageNet ResNet-18 (ONNX Model Zoo) and verify its checksum")
    pc = sub.add_parser("patchcore", help="train and evaluate PatchCore")
    pc.add_argument("--random-split", action="store_true", help="split by image (to measure leakage)")
    sub.add_parser("unet", help="train and evaluate the supervised U-Net")
    sub.add_parser("export", help="export the trained models to ONNX (models/)")
    bench = sub.add_parser("benchmark", help="ONNX Runtime latency on CPU")
    bench.add_argument("--runs", type=int, default=50)
    sub.add_parser("figures", help="generate the report figures")
    serve = sub.add_parser("serve", help="start the API server")
    serve.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("defectvision.api:app", host="0.0.0.0", port=args.port)
        return

    from . import experiments

    cfg = experiments.load_config(args.config)
    if args.cmd == "fetch-data":
        from .data import fetch

        print(fetch(cfg["data_dir"]))
    elif args.cmd == "fetch-weights":
        from .backbone import fetch_weights

        print(fetch_weights())
    elif args.cmd == "patchcore":
        _, report = experiments.run_patchcore(cfg, by_group=not args.random_split)
        print(json.dumps({k: v for k, v in report.items() if k != "split_summary"}, indent=2, ensure_ascii=False))
    elif args.cmd == "unet":
        _, report = experiments.run_unet(cfg)
        print(json.dumps({k: v for k, v in report.items() if k != "split_summary"}, indent=2, ensure_ascii=False))
    elif args.cmd == "export":
        from .release import export_all

        export_all(cfg)
    elif args.cmd == "benchmark":
        from .release import benchmark

        print(json.dumps(benchmark(cfg, runs=args.runs), indent=2))
    elif args.cmd == "figures":
        from .report import figures

        figures(cfg, Path("reports/figures"))


if __name__ == "__main__":
    main()
