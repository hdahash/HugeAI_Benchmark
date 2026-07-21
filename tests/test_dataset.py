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


def test_load_hugeai_dataset_shipped_with_repo():
    items = load_dataset("data/hugeai_prompts.jsonl")
    assert len(items) > 0

    multi_turn = [i for i in items if i.conversation]
    assert multi_turn, "expected at least one multi-turn item"
    for item in multi_turn:
        assert all("role" in turn and "content" in turn for turn in item.conversation)

    excludes_items = [i for i in items if i.expected_answer_excludes]
    assert excludes_items, "expected at least one item using expected_answer_excludes"


def test_load_dataset_with_conversation_and_excludes(tmp_path: Path):
    content = (
        '{"id": "mt-1", "prompt": "final turn", "conversation": '
        '[{"role": "user", "content": "first turn"}, {"role": "assistant", "content": "reply"}]}\n'
        '{"id": "sec-1", "prompt": "p", "expected_answer_excludes": ["forbidden"]}\n'
    )
    f = tmp_path / "data.jsonl"
    f.write_text(content)

    items = load_dataset(f)
    assert items[0].conversation == [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "reply"},
    ]
    assert items[1].expected_answer_excludes == ["forbidden"]
    assert items[1].conversation == []
