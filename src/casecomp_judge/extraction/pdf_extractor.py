"""PDF extraction for casecomp decks.

Casecomp decks are slide-style PDFs: every page is primarily visual
(frameworks, charts, matrices) with text that loses structure when
extracted naively. This module extracts, per slide:

  - layout-aware text (via pdfplumber)
  - a rasterized PNG of the full slide (via PyMuPDF), so a human or a
    future vision-capable model can inspect layout/charts directly

Output is a `DeckExtraction` dataclass that downstream agents consume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger("casecomp_judge.extraction")


@dataclass
class SlideExtraction:
    index: int  # 1-based slide/page number
    text: str
    image_path: str | None = None
    char_count: int = 0
    has_visual_content: bool = False  # vector drawings/images detected on page
    visual_object_count: int = 0  # raw count, for debugging/tuning the heuristic


@dataclass
class DeckExtraction:
    source_path: str
    deck_name: str
    slide_count: int
    slides: list[SlideExtraction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def full_text(self, max_chars_per_slide: int | None = None) -> str:
        """Concatenate all slide text into a single labeled block."""
        parts = []
        for slide in self.slides:
            text = slide.text
            if max_chars_per_slide and len(text) > max_chars_per_slide:
                text = text[:max_chars_per_slide] + " […truncated]"
            parts.append(f"--- Slide {slide.index} ---\n{text.strip()}")
        return "\n\n".join(parts)

    def merge_vision_descriptions(self, descriptions: dict[int, str]) -> None:
        """Append vision-model slide descriptions onto each slide's text.

        Mutates slides in place. Call this once, right after extraction,
        before handing the deck to the summarizer/judge agents — they
        only ever see `slide.text`, so this is the single integration
        point between the vision reader and the rest of the pipeline.
        """
        no_extra_marker = "no additional visual content"
        for slide in self.slides:
            desc = (descriptions.get(slide.index) or "").strip()
            if not desc or no_extra_marker in desc.lower():
                continue
            if slide.text.strip():
                slide.text = f"{slide.text.strip()}\n\n[Visual content on this slide]: {desc}"
            else:
                slide.text = f"[Visual content on this slide]: {desc}"
            slide.char_count = len(slide.text)


def extract_deck(
    pdf_path: str | Path,
    output_dir: str | Path,
    render_images: bool = True,
    dpi: int = 150,
) -> DeckExtraction:
    """Extract text and (optionally) rendered images for every slide.

    Args:
        pdf_path: path to the source PDF.
        output_dir: directory to write rendered slide images into
            (created if missing). Ignored if render_images=False.
        render_images: whether to rasterize each slide to PNG.
        dpi: resolution for rasterization.

    Returns:
        DeckExtraction with per-slide text and image paths.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = Path(output_dir)
    deck_name = pdf_path.stem
    warnings: list[str] = []

    slides: list[SlideExtraction] = []

    # --- Text extraction (pdfplumber: layout-aware) ---
    text_by_page: dict[int, str] = {}
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:  # noqa: BLE001 - keep extraction resilient
                    logger.warning("Text extraction failed on slide %d: %s", i, exc)
                    text = ""
                    warnings.append(f"Slide {i}: text extraction failed ({exc})")
                text_by_page[i] = text
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to open PDF with pdfplumber: {exc}") from exc

    slide_count = len(text_by_page)
    if slide_count == 0:
        warnings.append("No pages found in PDF.")

    # Detect likely-scanned slides (no extractable text at all)
    empty_slides = [i for i, t in text_by_page.items() if not t.strip()]
    if empty_slides and len(empty_slides) == slide_count:
        warnings.append(
            "No text layer detected on ANY slide — this PDF may be scanned/"
            "image-only. Rendered images will still be produced; consider "
            "enabling a vision-capable model (extraction.use_vision: true) "
            "or running OCR."
        )
    elif empty_slides:
        warnings.append(
            f"Slides with no extractable text (possibly image-only): "
            f"{empty_slides}"
        )

    # --- Image rendering (PyMuPDF) + visual-content heuristic ---
    # We detect whether a page likely has chart/diagram content by counting
    # vector drawing objects (rects, lines, curves — what charts/frameworks
    # are made of) and embedded raster images, directly from the PDF's
    # object model. This is a cheap, deterministic signal computed once
    # during extraction — used later to decide whether running an LLM
    # vision pass on a given slide is actually worth the call, instead of
    # always running vision on every slide regardless of content.
    image_paths: dict[int, str] = {}
    visual_signal: dict[int, tuple[bool, int]] = {}
    if render_images:
        images_dir = output_dir / "slides"
        images_dir.mkdir(parents=True, exist_ok=True)
        try:
            doc = fitz.open(str(pdf_path))
            zoom = dpi / 72.0  # PDF default is 72 DPI
            matrix = fitz.Matrix(zoom, zoom)
            for i, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=matrix)
                img_path = images_dir / f"slide_{i:03d}.png"
                pix.save(str(img_path))
                image_paths[i] = str(img_path)

                try:
                    drawing_count = len(page.get_drawings())
                    embedded_image_count = len(page.get_images())
                except Exception:  # noqa: BLE001
                    drawing_count, embedded_image_count = 0, 0
                visual_object_count = drawing_count + embedded_image_count
                # A handful of drawings is often just a border/divider line;
                # charts/diagrams/frameworks tend to involve many more
                # shapes (bars, gridlines, axis ticks, boxes, connectors).
                has_visual = visual_object_count >= 4 or embedded_image_count >= 1
                visual_signal[i] = (has_visual, visual_object_count)
            doc.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Image rendering failed: %s", exc)
            warnings.append(f"Image rendering failed: {exc}")

    for i in range(1, slide_count + 1):
        text = text_by_page.get(i, "")
        has_visual, visual_count = visual_signal.get(i, (False, 0))
        slides.append(
            SlideExtraction(
                index=i,
                text=text,
                image_path=image_paths.get(i),
                char_count=len(text),
                has_visual_content=has_visual,
                visual_object_count=visual_count,
            )
        )

    extraction = DeckExtraction(
        source_path=str(pdf_path),
        deck_name=deck_name,
        slide_count=slide_count,
        slides=slides,
        warnings=warnings,
    )

    logger.info(
        "Extracted '%s': %d slides, %d with images, %d warnings",
        deck_name,
        slide_count,
        len(image_paths),
        len(warnings),
    )
    return extraction
