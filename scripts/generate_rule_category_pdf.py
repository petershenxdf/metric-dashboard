from __future__ import annotations

import os
import re
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex-cache")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "rule_interpretation_categories_guide.md"
OUTPUT = ROOT / "docs" / "rule_interpretation_categories_guide.pdf"
FONT_PATHS = (
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)


PAGE_W = 8.27
PAGE_H = 11.69
LEFT = 0.72
RIGHT = 0.62
TOP = 0.68
BOTTOM = 0.68
CONTENT_W = PAGE_W - LEFT - RIGHT


def main() -> int:
    font = _font()
    regular = font
    bold = font
    source = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SOURCE
    output = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else OUTPUT
    lines = source.read_text(encoding="utf-8").splitlines()

    with PdfPages(output) as pdf:
        writer = PdfWriter(pdf, regular, bold)
        for raw_line in lines:
            writer.write_markdown_line(raw_line)
        writer.finish()

    print(output)
    return 0


class PdfWriter:
    def __init__(self, pdf: PdfPages, regular: FontProperties, bold: FontProperties):
        self.pdf = pdf
        self.regular = regular
        self.bold = bold
        self.page_no = 0
        self.fig = None
        self.y = 0.0
        self.new_page()

    def new_page(self) -> None:
        if self.fig is not None:
            self._footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.page_no += 1
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor="white")
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.y = PAGE_H - TOP

    def finish(self) -> None:
        self._footer()
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def write_markdown_line(self, raw_line: str) -> None:
        line = raw_line.rstrip()
        if not line:
            self._space(0.12)
            return

        if line.startswith("## ") and re.match(r"## \d+\. ", line):
            if self.y < PAGE_H - TOP - 0.2:
                self.new_page()

        if line.startswith("# "):
            self._text(line[2:], size=24, font=self.bold, color="#1f2937", gap_before=0.08, gap_after=0.22, width=30)
        elif line.startswith("## "):
            self._text(line[3:], size=17, font=self.bold, color="#1f2937", gap_before=0.16, gap_after=0.12, width=42)
        elif line.startswith("### "):
            self._text(line[4:], size=12.5, font=self.bold, color="#1f2937", gap_before=0.14, gap_after=0.05, width=54)
        elif line.startswith("- "):
            self._bullet(line[2:])
        elif re.match(r"\d+\. ", line):
            self._numbered(line)
        else:
            self._text(line, size=10.2, font=self.regular, color="#202124", gap_before=0.02, gap_after=0.05, width=72)

    def _text(
        self,
        text: str,
        *,
        size: float,
        font: FontProperties,
        color: str,
        gap_before: float,
        gap_after: float,
        width: int,
        x: float = LEFT,
        line_height: float | None = None,
    ) -> None:
        self._space(gap_before)
        wrapped = _wrap(text, width)
        line_height = line_height or size / 72 * 1.48
        self._ensure(line_height * len(wrapped) + gap_after)
        for wrapped_line in wrapped:
            self.fig.text(
                x / PAGE_W,
                self.y / PAGE_H,
                wrapped_line,
                ha="left",
                va="top",
                color=color,
                fontsize=size,
                fontproperties=font,
            )
            self.y -= line_height
        self._space(gap_after)

    def _bullet(self, text: str) -> None:
        wrapped = _wrap(text, 68)
        line_height = 10.0 / 72 * 1.45
        self._ensure(line_height * len(wrapped) + 0.04)
        self.fig.text(LEFT / PAGE_W, self.y / PAGE_H, "•", ha="left", va="top", fontsize=10.0, fontproperties=self.bold)
        for index, wrapped_line in enumerate(wrapped):
            self.fig.text(
                (LEFT + 0.22) / PAGE_W,
                self.y / PAGE_H,
                wrapped_line,
                ha="left",
                va="top",
                fontsize=10.0,
                color="#202124",
                fontproperties=self.regular,
            )
            self.y -= line_height
        self._space(0.035)

    def _numbered(self, text: str) -> None:
        number, body = text.split(" ", 1)
        wrapped = _wrap(body, 68)
        line_height = 10.0 / 72 * 1.45
        self._ensure(line_height * len(wrapped) + 0.04)
        self.fig.text(LEFT / PAGE_W, self.y / PAGE_H, number, ha="left", va="top", fontsize=10.0, fontproperties=self.bold)
        for wrapped_line in wrapped:
            self.fig.text(
                (LEFT + 0.28) / PAGE_W,
                self.y / PAGE_H,
                wrapped_line,
                ha="left",
                va="top",
                fontsize=10.0,
                color="#202124",
                fontproperties=self.regular,
            )
            self.y -= line_height
        self._space(0.035)

    def _space(self, amount: float) -> None:
        self.y -= amount

    def _ensure(self, height: float) -> None:
        if self.y - height < BOTTOM:
            self.new_page()

    def _footer(self) -> None:
        self.fig.text(
            0.5,
            0.035,
            f"Rule Interpretation Categories Guide · page {self.page_no}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#6b7280",
            fontproperties=self.regular,
        )


def _wrap(text: str, width: int) -> list[str]:
    cleaned = text.replace("`", "")
    return textwrap.wrap(
        cleaned,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
    ) or [""]


def _font() -> FontProperties:
    for path in FONT_PATHS:
        if Path(path).exists():
            return FontProperties(fname=path)
    raise RuntimeError("No compatible CJK font found on this macOS system.")


if __name__ == "__main__":
    raise SystemExit(main())
