from app.core.chunking import chunk_text


def test_chunking_basic():
    text = "a" * 1200
    chunks = chunk_text(text, "doc.txt", chunk_size=500, overlap=80)
    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "doc.txt::0"
    assert all(chunk.source_file == "doc.txt" for chunk in chunks)
