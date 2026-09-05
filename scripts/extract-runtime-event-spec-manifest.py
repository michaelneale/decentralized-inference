#!/usr/bin/env python3
"""Extract section 8 required-event bullets from the event-system spec."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from typing import Final


FAMILY_PATTERN: Final = re.compile(r"^### (8\.(?:[1-9]|1[0-5])) (.+)$")
REQUIRED_LABELS: Final = frozenset(
    {"Required events:", "Required derived state events:", "Required events or counters:"}
)


@dataclass(frozen=True)
class SpecBullet:
    section: str
    family: str
    ordinal: int
    text: str


class SpecManifestError(RuntimeError):
    pass


def extract_bullets(spec_text: str) -> tuple[SpecBullet, ...]:
    bullets: list[SpecBullet] = []
    section = ""
    family = ""
    collecting = False
    current: list[str] = []
    ordinal = 0

    def finish_bullet() -> None:
        nonlocal current
        if current:
            bullets.append(SpecBullet(section, family, ordinal, " ".join(current)))
            current = []

    for line in spec_text.splitlines():
        if line.startswith("## 9."):
            finish_bullet()
            break
        family_match = FAMILY_PATTERN.fullmatch(line)
        if family_match is not None:
            finish_bullet()
            section, family = family_match.groups()
            collecting = False
            ordinal = 0
            continue
        if line in REQUIRED_LABELS:
            if not section:
                raise SpecManifestError("required-event list appears outside section 8 family")
            collecting = True
            continue
        if not collecting:
            continue
        if line.startswith("- "):
            finish_bullet()
            ordinal += 1
            current = [line[2:].strip()]
            continue
        if current and line.startswith("  "):
            current.append(line.strip())
            continue
        if line:
            finish_bullet()
            collecting = False

    sections = {bullet.section for bullet in bullets}
    expected_sections = {f"8.{index}" for index in range(1, 16)}
    if sections != expected_sections:
        raise SpecManifestError(
            f"section 8 family mismatch: missing={sorted(expected_sections - sections)}, "
            f"unexpected={sorted(sections - expected_sections)}"
        )
    return tuple(bullets)


def render_manifest(bullets: tuple[SpecBullet, ...]) -> str:
    payload = {
        "schema_version": 1,
        "source": ".omo/specs/event-system.md",
        "bullet_count": len(bullets),
        "bullets": [asdict(bullet) for bullet in bullets],
    }
    return json.dumps(payload, indent=2, ensure_ascii=True) + "\n"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=repo_root / ".omo/specs/event-system.md"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root
        / "crates/mesh-llm-runtime-event-contracts/inventory/spec_manifest.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rendered = render_manifest(extract_bullets(args.spec.read_text(encoding="utf-8")))
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current == rendered:
            return 0
        print(f"stale generated spec manifest: {args.output}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"generated {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
