"""
Section-based text chunker with metadata preservation.
Splits plain-text documents at ## headings, keeping section title
and source metadata with each chunk for citation purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str       # "{source_id}__{section_index}"
    source_id: str
    title: str          # document title
    section: str        # section heading
    text: str           # chunk content (heading + body)
    crops: list[str]
    diseases: list[str]
    regions: list[str]
    publisher: str


def chunk_document(
    text: str,
    source_id: str,
    title: str,
    crops: list[str],
    diseases: list[str],
    regions: list[str],
    publisher: str,
    min_chars: int = 100,
) -> list[Chunk]:
    """Splits a document at ## headings. Each chunk = one section.
    Sections shorter than min_chars are merged into the previous chunk."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = title
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = line[3:].strip()
            current_lines = []
        elif line.startswith("# "):
            # Document title line — skip, already in metadata
            continue
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    chunks: list[Chunk] = []
    for i, (heading, body_lines) in enumerate(sections):
        body = "\n".join(body_lines).strip()
        if not body:
            continue

        # Merge very short sections into previous chunk
        if len(body) < min_chars and chunks:
            prev = chunks[-1]
            merged_text = prev.text + "\n\n" + heading + "\n" + body
            chunks[-1] = Chunk(
                chunk_id=prev.chunk_id,
                source_id=prev.source_id,
                title=prev.title,
                section=prev.section,
                text=merged_text,
                crops=prev.crops,
                diseases=prev.diseases,
                regions=prev.regions,
                publisher=prev.publisher,
            )
            continue

        chunks.append(Chunk(
            chunk_id=f"{source_id}__{i}",
            source_id=source_id,
            title=title,
            section=heading,
            text=f"{heading}\n\n{body}",
            crops=crops,
            diseases=diseases,
            regions=regions,
            publisher=publisher,
        ))

    return chunks


def chunk_source(source_meta: dict, sources_dir: Path) -> list[Chunk]:
    """Chunks one source entry from manifest.yaml."""
    file_path = sources_dir.parent / source_meta["file"]
    text = file_path.read_text(encoding="utf-8")
    return chunk_document(
        text=text,
        source_id=source_meta["id"],
        title=source_meta["title"],
        crops=source_meta["crops"],
        diseases=source_meta["diseases"],
        regions=source_meta["regions"],
        publisher=source_meta["publisher"],
    )