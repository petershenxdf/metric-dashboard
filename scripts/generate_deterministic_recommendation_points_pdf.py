from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "deterministic_recommendation_points_guide.md"
OUTPUT = ROOT / "docs" / "deterministic_recommendation_points_guide.pdf"
TITLE = "Deterministic Recommendation Points Guide"
WIDTH = 88


def main() -> int:
    source = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SOURCE
    output = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else OUTPUT
    text = _markdown_to_print_text(source.read_text(encoding="utf-8"))

    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as handle:
        handle.write(text)
        text_path = Path(handle.name)

    try:
        with output.open("wb") as pdf_handle:
            subprocess.run(
                ["cupsfilter", "-m", "application/pdf", str(text_path)],
                check=True,
                stdout=pdf_handle,
            )
    finally:
        text_path.unlink(missing_ok=True)

    print(output)
    return 0


def _markdown_to_print_text(markdown: str) -> str:
    lines: list[str] = []
    in_code = False

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            in_code = not in_code
            lines.append("")
            continue

        if not line:
            lines.append("")
            continue

        if in_code:
            lines.extend(_wrap("    " + line, WIDTH))
            continue

        if line.startswith("# "):
            heading = _clean(line[2:]).upper()
            lines.extend([heading, "=" * min(len(heading), WIDTH), ""])
            continue

        if line.startswith("## "):
            heading = _clean(line[3:])
            lines.extend(["", heading, "-" * min(len(heading), WIDTH)])
            continue

        if line.startswith("### "):
            lines.extend(["", _clean(line[4:]) + ":"])
            continue

        if line.startswith("- "):
            body = _clean(line[2:])
            wrapped = _wrap(body, WIDTH - 4)
            for index, wrapped_line in enumerate(wrapped):
                prefix = "- " if index == 0 else "  "
                lines.append(prefix + wrapped_line)
            continue

        if re.match(r"\d+\. ", line):
            number, body = line.split(" ", 1)
            body = _clean(body)
            wrapped = _wrap(body, WIDTH - len(number) - 2)
            for index, wrapped_line in enumerate(wrapped):
                prefix = f"{number} " if index == 0 else " " * (len(number) + 1)
                lines.append(prefix + wrapped_line)
            continue

        lines.extend(_wrap(_clean(line), WIDTH))

    return "\n".join([TITLE, "", *lines, ""])


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=False,
    ) or [""]


def _clean(text: str) -> str:
    return text.replace("`", "").replace("**", "")


if __name__ == "__main__":
    raise SystemExit(main())
