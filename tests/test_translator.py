import json
from unittest.mock import MagicMock, patch

from src.translator import extract_code_blocks, restore_code_blocks, translate, translate_many, _cache_key


def test_extract_code_blocks_finds_repl_lines():
    text = "Explanation.\n>>> df.head()\n>>> df.shape\nMore text."
    stripped, blocks = extract_code_blocks(text)

    assert "<CODE_BLOCK_0>" in stripped
    assert len(blocks) == 1
    assert ">>> df.head()" in blocks[0]
    assert ">>> df.shape" in blocks[0]


def test_extract_code_blocks_finds_jupyter_cells():
    text = "Intro.\nIn [1]: x = 1\nIn [2]: print(x)\nOutro."
    stripped, blocks = extract_code_blocks(text)

    assert "<CODE_BLOCK_0>" in stripped
    assert "In [1]: x = 1" in blocks[0]


def test_extract_code_blocks_finds_indented_blocks():
    text = "Before.\n    x = 1\n    y = 2\n    z = x + y\nAfter."
    stripped, blocks = extract_code_blocks(text)

    assert "<CODE_BLOCK_0>" in stripped
    assert "x = 1" in blocks[0]


def test_restore_code_blocks_reinserts_original():
    original = "Text.\n>>> a = 1\nMore text."
    stripped, blocks = extract_code_blocks(original)
    restored = restore_code_blocks(stripped, blocks)

    assert restored == original


def test_extract_returns_text_unchanged_when_no_code():
    text = "Just regular prose without code."
    stripped, blocks = extract_code_blocks(text)

    assert stripped == text
    assert blocks == []


@patch("src.translator.groq_client")
def test_translate_sends_glossary_and_code_preserved(mock_client):
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Isto é um DataFrame.\n<CODE_BLOCK_0>"))]
    )

    english = "This is a DataFrame.\n>>> df.head()"
    result = translate(english)

    # result should have original code block restored
    assert ">>> df.head()" in result
    # DataFrame kept in English
    assert "DataFrame" in result

    # glossary should have been included in system prompt
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    system_msg = call_kwargs["messages"][0]["content"]
    assert "DataFrame" in system_msg
    assert "NÃO traduza" in system_msg


@patch("src.translator.groq_client")
def test_translate_strips_preamble(mock_client):
    """If Groq echoes preambles like 'Tradução:', they get stripped."""
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Tradução:\nTexto traduzido."))]
    )
    result = translate("Some text.")
    assert result == "Texto traduzido."


def test_cache_key_is_stable():
    a = _cache_key("hello world")
    b = _cache_key("hello world")
    c = _cache_key("different")
    assert a == b
    assert a != c


@patch("src.translator.translate")
def test_translate_many_uses_cache(mock_translate, tmp_path):
    mock_translate.side_effect = lambda t: f"PT[{t}]"
    cache_path = tmp_path / "cache.jsonl"

    texts = ["one", "two", "one"]  # "one" repeats
    results = translate_many(texts, cache_path=cache_path)

    assert results == ["PT[one]", "PT[two]", "PT[one]"]
    # translate() called only twice (once per unique input)
    assert mock_translate.call_count == 2


@patch("src.translator.translate")
def test_translate_many_persists_cache(mock_translate, tmp_path):
    mock_translate.side_effect = lambda t: f"PT[{t}]"
    cache_path = tmp_path / "cache.jsonl"

    translate_many(["alpha"], cache_path=cache_path)

    # second call in a fresh process — mock must NOT be called again
    mock_translate.reset_mock()
    results = translate_many(["alpha"], cache_path=cache_path)
    assert results == ["PT[alpha]"]
    assert mock_translate.call_count == 0
