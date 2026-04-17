from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_viewer.config import load_model_config
from llm_viewer.profiles import ProfileName, get_profile
from llm_viewer.registry import build_graph_bundle
from llm_viewer.server import run_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-viewer")
    subparsers = parser.add_subparsers(dest="command")

    extract = subparsers.add_parser("extract", help="Generate a graph bundle from config.json")
    extract.add_argument("config", help="Path to local config.json")
    extract.add_argument(
        "--profile",
        choices=[profile.value for profile in ProfileName],
        default=ProfileName.PREFILL.value,
        help="Runtime profile to simulate",
    )
    extract.add_argument("--output", help="Output file path. Defaults to stdout")
    extract.add_argument("--indent", type=int, default=2, help="JSON indent")

    app = subparsers.add_parser("app", help="Start local browser app")
    app.add_argument("--host", default="127.0.0.1", help="Host to bind")
    app.add_argument("--port", type=int, default=8000, help="Port to bind")
    app.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        return run_app(host="127.0.0.1", port=8000, open_browser=True)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "extract":
        config = load_model_config(args.config)
        profile = get_profile(ProfileName(args.profile))
        bundle = build_graph_bundle(config=config, profile=profile)
        payload = json.dumps(bundle.to_dict(), indent=args.indent, ensure_ascii=False)

        if args.output:
            Path(args.output).write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return 0

    if args.command == "app":
        return run_app(host=args.host, port=args.port, open_browser=not args.no_open)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
