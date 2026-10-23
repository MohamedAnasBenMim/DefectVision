"""Ligne de commande : `dv <commande>` (voir README)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dv", description="DefectVision")
    p.add_argument("--config", default="configs/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-data", help="télécharge le jeu de données (commit figé)")
    sub.add_parser("fetch-weights", help="télécharge ResNet-18 ImageNet (ONNX Model Zoo) et vérifie l'empreinte")
    pc = sub.add_parser("patchcore", help="entraîne et évalue PatchCore")
    pc.add_argument("--random-split", action="store_true", help="découpage par image (pour mesurer la fuite)")
    sub.add_parser("unet", help="entraîne et évalue le U-Net supervisé")
    sub.add_parser("export", help="exporte les modèles entraînés en ONNX (models/)")
    bench = sub.add_parser("benchmark", help="latence ONNX Runtime sur CPU")
    bench.add_argument("--runs", type=int, default=50)
    sub.add_parser("figures", help="génère les figures du rapport")
    serve = sub.add_parser("serve", help="lance l'API")
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
