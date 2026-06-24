
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import fitz  # PyMuPDF
import pytest

from casecomp_judge.extraction.pdf_extractor import extract_deck


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Builds a tiny 2-page PDF with simple text, for fast offline tests."""
    pdf_path = tmp_path / "sample_deck.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Slide 1: Problem Statement")
    page1.insert_text((72, 100), "Our client faces declining margins.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Slide 2: Recommendation")
    page2.insert_text((72, 100), "We recommend a pricing overhaul.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_extract_deck_basic(sample_pdf: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    deck = extract_deck(sample_pdf, output_dir=output_dir, render_images=True)

    assert deck.slide_count == 2
    assert len(deck.slides) == 2
    assert "Problem Statement" in deck.slides[0].text
    assert "Recommendation" in deck.slides[1].text
    assert deck.slides[0].image_path is not None
    assert Path(deck.slides[0].image_path).exists()


def test_extract_deck_no_images(sample_pdf: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "output_no_images"
    deck = extract_deck(sample_pdf, output_dir=output_dir, render_images=False)

    assert deck.slide_count == 2
    assert all(s.image_path is None for s in deck.slides)


def test_full_text_concatenation(sample_pdf: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "output2"
    deck = extract_deck(sample_pdf, output_dir=output_dir, render_images=False)
    full_text = deck.full_text()

    assert "--- Slide 1 ---" in full_text
    assert "--- Slide 2 ---" in full_text


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_deck(tmp_path / "does_not_exist.pdf", output_dir=tmp_path)


def test_merge_vision_descriptions(sample_pdf: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "output3"
    deck = extract_deck(sample_pdf, output_dir=output_dir, render_images=False)

    original_slide1_text = deck.slides[0].text
    descriptions = {
        1: "No additional visual content beyond the extracted text.",
        2: "A bar chart showing rising values across four categories.",
    }
    deck.merge_vision_descriptions(descriptions)

    # Slide 1: "no additional content" marker should NOT be appended
    assert deck.slides[0].text == original_slide1_text
    assert "[Visual content" not in deck.slides[0].text

    # Slide 2: real visual description SHOULD be appended
    assert "[Visual content on this slide]:" in deck.slides[1].text
    assert "bar chart" in deck.slides[1].text


def test_merge_vision_descriptions_empty_dict_is_noop(
    sample_pdf: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "output4"
    deck = extract_deck(sample_pdf, output_dir=output_dir, render_images=False)
    original_texts = [s.text for s in deck.slides]

    deck.merge_vision_descriptions({})

    assert [s.text for s in deck.slides] == original_texts

