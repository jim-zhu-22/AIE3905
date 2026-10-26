"""CLI contract example; returns an explicit unknown instead of invented labels."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = {"status": "unknown", "source": "physical_template", "reason": "物理属性算法尚未接入",
              "values": {"instance_id": request["instance_id"]}}
    args.response.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
