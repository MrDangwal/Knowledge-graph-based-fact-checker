from app.core.text_utils import split_sentences_with_offsets, split_claims_with_offsets


def test_split_sentences_with_offsets():
    text = "Hello world. This is a test! Another sentence?"
    spans = split_sentences_with_offsets(text)
    assert len(spans) == 3
    assert spans[0].text == "Hello world."
    assert text[spans[0].start:spans[0].end] == spans[0].text
    assert spans[1].text == "This is a test!"
    assert spans[2].text == "Another sentence?"


def test_split_claims_with_offsets():
    text = "Paris is the capital of France, and Berlin is the capital of Germany."
    spans = split_claims_with_offsets(text)
    assert len(spans) == 2
    assert spans[0].text == "Paris is the capital of France"
    assert spans[1].text == "and Berlin is the capital of Germany"
