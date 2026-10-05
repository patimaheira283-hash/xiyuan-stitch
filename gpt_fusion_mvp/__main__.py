import argparse
import json
import sys
from pathlib import Path

from .core import LIBRARY, MODEL, MAINLINE_MODEL, PROMPT, helper_path, load_provider, run_fusion
from .materials import build_library, get_case


def main():
    parser = argparse.ArgumentParser(description="两张原图直接交给 GPT Image 2.5 融合")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("materials")
    sub.add_parser("doctor")
    run = sub.add_parser("run")
    run.add_argument("--case", default="coffee-clean")
    run.add_argument("--left", type=Path)
    run.add_argument("--right", type=Path)
    run.add_argument("--model", default=MODEL)
    run.add_argument("--mainline-model", default=MAINLINE_MODEL)
    run.add_argument("--quality", default="high")
    run.add_argument("--provider-id")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--route", choices=("responses", "images"), default="responses")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=18767)
    serve.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.command == "materials":
        m = build_library()
        result = {"sources": len(m["sources"]), "cases": len(m["cases"]), "folder": str(LIBRARY)}
    elif args.command == "doctor":
        result = {"provider": load_provider().public(), "helper_available": helper_path().is_file(), "model": MODEL}
    elif args.command == "serve":
        from .server import serve
        serve(args.port, args.open)
        return
    else:
        prompt = PROMPT
        if args.left or args.right:
            if not (args.left and args.right):
                parser.error("--left 和 --right 必须同时提供。")
            left, right, truth, cid = args.left, args.right, None, "custom"
        else:
            case = get_case(args.case)
            left, right, truth, cid = LIBRARY / case["left"], LIBRARY / case["right"], LIBRARY / case["reference"] if case.get("reference") else None, case["id"]
            prompt = case.get("prompt", PROMPT)
        result = run_fusion(left, right, truth=truth, prompt=prompt, case_id=cid, model=args.model, quality=args.quality,
                            provider_id=args.provider_id, dry_run=args.dry_run, route=args.route, mainline_model=args.mainline_model)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
