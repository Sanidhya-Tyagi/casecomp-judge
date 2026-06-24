"""Vision reader.

When a vision-capable Ollama model is configured (extraction.use_vision:
true in config.yaml), this module looks at each slide's rendered image
and produces a rich text description of everything the text layer
alone would miss: charts, diagrams, frameworks (2x2s, funnels, value
chains), tables, icons, and visual emphasis.

Routing, not blanket application
---------------------------------
Vision calls aren't run unconditionally on every slide. Each slide
already carries a cheap, deterministic `has_visual_content` signal
computed during extraction (vector-drawing and embedded-image counts
from the PDF object model — see `extraction/pdf_extractor.py`). Slides
flagged as plain text skip the vision call entirely: there's nothing a
vision pass would recover that the text layer hasn't already captured,
so spending an LLM call on it is pure latency with no judging benefit.
This is the orchestrator deciding *which tool to use, per slide*,
rather than a hardcoded "always do X then Y" sequence.

Design choice: one image per Ollama call. Not all local vision models
reliably support multiple images in a single request, so per-slide
calls are the robust default — slower, but works across llava,
llama3.2-vision, qwen2.5vl, etc. without surprises.

The output is plain text, which is then merged with the extracted text
layer before being handed to the summarizer/judge agents — so those
agents stay text-only consumers and don't need to know whether vision
was used.
"""

from __future__ import annotations

import logging

from casecomp_judge.extraction.pdf_extractor import DeckExtraction, SlideExtraction
from casecomp_judge.utils.ollama_client import OllamaClient, OllamaError

logger = logging.getLogger("casecomp_judge.agents.vision_reader")


VISION_SYSTEM_PROMPT = """\
You are a meticulous visual analyst reviewing a single slide from a business \
case competition presentation. Describe everything relevant to judging the \
slide's analytical content. Be factual and specific — describe what is \
actually shown, including exact numbers, labels, and axis values where \
legible. Do not invent data you cannot read. If text is too small or blurry \
to read, say so rather than guessing.
"""

VISION_PROMPT_TEMPLATE = """\
This is slide {index} of a case competition deck titled "{deck_name}".

The text layer extracted from this slide (may be incomplete or jumbled, \
especially for diagrams/charts) is:
---
{extracted_text}
---

Look at the slide image and describe:
1. Any charts, graphs, or plots — type, axes, key data points/trends, and \
   what conclusion the slide seems to draw from them.
2. Any frameworks or diagrams (2x2 matrices, funnels, process flows, value \
   chains, etc.) — their structure and the content placed within them.
3. Any tables — key rows/columns and notable figures.
4. Any other visual content (icons, images, logos) only if it carries \
   analytical meaning (e.g. a competitor logo in a positioning map).
5. Overall visual emphasis — what is the slide's main visual takeaway?

If the slide is plain text with no meaningful charts/diagrams/tables beyond \
what's already in the extracted text layer, simply respond: "No additional \
visual content beyond the extracted text."

Keep your response focused and factual, under 200 words.
"""


def describe_slide_visually(
    client: OllamaClient, slide: SlideExtraction, deck_name: str
) -> str:
    """Get a text description of one slide's visual content from a vision model.

    Returns an empty string (rather than raising) if the slide has no
    rendered image, or if the vision call fails — the caller should
    treat this as "no additional visual info" and fall back to text-only.
    """
    if not slide.image_path:
        return ""

    prompt = VISION_PROMPT_TEMPLATE.format(
        index=slide.index,
        deck_name=deck_name,
        extracted_text=slide.text.strip() or "(no text extracted)",
    )

    try:
        response = client.generate_with_image(
            prompt=prompt, image_path=slide.image_path, system=VISION_SYSTEM_PROMPT
        )
        return response.text.strip()
    except OllamaError as exc:
        logger.warning(
            "Vision read failed for slide %d (%s); continuing without it: %s",
            slide.index,
            deck_name,
            exc,
        )
        return ""


def build_vision_enriched_text(
    client: OllamaClient,
    deck: DeckExtraction,
    max_slides: int | None = None,
    force_all: bool = False,
) -> dict[int, str]:
    """Run the vision reader over slides in a deck that warrant it.

    Routing: by default (force_all=False), a slide is only sent to the
    vision model if `slide.has_visual_content` is True — i.e. the PDF
    object model shows enough vector drawings or embedded images to
    suggest a chart, diagram, table, or framework. Slides that are
    plain text skip the vision call entirely; the text layer already
    captures everything there is to capture, so spending a model call
    there is pure latency with no judging upside.

    Set force_all=True to bypass routing and vision-read every slide
    regardless of the heuristic (useful if you suspect the heuristic
    is wrong for a particular deck, e.g. unusual PDF export pipeline).

    Returns a dict of {slide_index: visual_description}. Skipped slides
    are included with an empty string so callers can distinguish
    "skipped" from "vision call made but found nothing."
    """
    slides = deck.slides if max_slides is None else deck.slides[:max_slides]
    descriptions: dict[int, str] = {}
    skipped = 0

    for slide in slides:
        if not force_all and not slide.has_visual_content:
            descriptions[slide.index] = ""
            skipped += 1
            continue

        logger.info(
            "Vision-reading slide %d/%d of '%s' (visual_object_count=%d)...",
            slide.index,
            deck.slide_count,
            deck.deck_name,
            slide.visual_object_count,
        )
        descriptions[slide.index] = describe_slide_visually(
            client, slide, deck.deck_name
        )

    if skipped:
        logger.info(
            "Vision routing: skipped %d/%d slide(s) on '%s' with no "
            "detected visual content (text-only, no LLM call made).",
            skipped,
            len(slides),
            deck.deck_name,
        )
    return descriptions
