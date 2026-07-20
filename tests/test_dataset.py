from pathlib import Path

from routerbench.dataset import load_dataset


def test_load_dataset(tmp_path: Path):
    content = (
        '{"id": "a", "prompt": "hello", "category": "greet", "expected_model": "small-model", "expected_answer_contains": ["hi"]}\n'
        "\n"
        "# a comment line\n"
        '{"id": "b", "prompt": "world"}\n'
    )
    f = tmp_path / "data.jsonl"
    f.write_text(content)

    items = load_dataset(f)
    assert len(items) == 2
    assert items[0].id == "a"
    assert items[0].category == "greet"
    assert items[0].expected_model == "small-model"
    assert items[0].expected_answer_contains == ["hi"]
    assert items[1].category == "general"
    assert items[1].expected_model is None


def test_load_sample_dataset_shipped_with_repo():
    items = load_dataset("data/sample_prompts.jsonl")
    assert len(items) > 0
    assert all(item.prompt for item in items)
