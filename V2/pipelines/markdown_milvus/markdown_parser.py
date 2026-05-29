from __future__ import annotations

import re
from pathlib import Path

from .models import MarkdownBlock, RawReport


FRONTMATTER_BOUNDARY = "---"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")


def load_raw_reports(raw_dir: Path) -> list[RawReport]:
    reports: list[RawReport] = []
    for path in sorted(raw_dir.glob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        metadata, _ = parse_frontmatter(markdown)
        reports.append(
            RawReport(
                path=path,
                company=required_metadata(metadata, "company", path),
                ticker=required_metadata(metadata, "ticker", path),
                year=int(required_metadata(metadata, "year", path)),
                markdown=markdown,
            )
        )
    return reports


def parse_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_BOUNDARY:
        return {}, markdown

    metadata: dict[str, str] = {}
    body_start = 0
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == FRONTMATTER_BOUNDARY:
            body_start = index + 1
            break
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    body = "\n".join(lines[body_start:])
    return metadata, body


def required_metadata(metadata: dict[str, str], key: str, path: Path) -> str:
    value = metadata.get(key)
    if not value:
        raise ValueError(f"{path} missing required frontmatter field: {key}")
    return value


def parse_markdown_report(report: RawReport) -> list[MarkdownBlock]:
    _, body = parse_frontmatter(report.markdown)
    lines = body.splitlines()
    blocks: list[MarkdownBlock] = []
    paragraph: list[str] = []
    table: list[str] = []
    heading = ""
    order = 0

    def next_order() -> int:
        nonlocal order
        order += 1
        return order

    def flush_paragraph() -> None:
        if not paragraph:
            return
        text = " ".join(part.strip() for part in paragraph if part.strip())
        paragraph.clear()
        if text:
            blocks.append(MarkdownBlock("text", heading, text, next_order()))

    def flush_table() -> None:
        if not table:
            return
        text = "\n".join(table)
        table.clear()
        blocks.append(MarkdownBlock("table", heading, text, next_order()))

    for line in lines:
        stripped = line.strip()
        heading_match = HEADING_RE.match(stripped)
        image_match = IMAGE_RE.search(stripped)
        is_table_line = stripped.startswith("|") and stripped.endswith("|")

        if heading_match:
            flush_paragraph()
            flush_table()
            heading = heading_match.group(2).strip()
            continue

        if not stripped:
            flush_paragraph()
            flush_table()
            continue

        if image_match:
            flush_paragraph()
            flush_table()
            blocks.append(MarkdownBlock("figure", heading, figure_raw_text(image_match), next_order()))
            continue

        if is_table_line:
            flush_paragraph()
            table.append(stripped)
            continue

        flush_table()
        paragraph.append(stripped)

    flush_paragraph()
    flush_table()
    return blocks


def figure_raw_text(match: re.Match[str]) -> str:
    path = match.group("path").strip()
    alt = match.group("alt").strip()
    if not alt:
        return path
    return f"{path}\nalt: {alt}"
