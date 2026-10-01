from pathlib import Path

from signworld.acquisition.ledger import Ledger, read_ledgers
from signworld.acquisition.outcome import Outcome, Status
from signworld.acquisition.sharding import Shard


def test_latest_outcome_wins_and_attempts_accumulate(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "shard.jsonl")
    ledger.record(Outcome("a", Status.FAILED, "timeout"))
    ledger.record(Outcome("a", Status.DONE))

    entry = Ledger(tmp_path / "shard.jsonl").latest()["a"]

    assert entry.status is Status.DONE
    assert entry.attempts == 2


def test_refusals_do_not_count_as_attempts(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "shard.jsonl")
    ledger.record(Outcome("a", Status.BLOCKED))
    ledger.record(Outcome("a", Status.BLOCKED))

    assert ledger.record(Outcome("a", Status.FAILED)).attempts == 1


def test_items_failing_too_often_are_settled(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "shard.jsonl")
    ledger.record(Outcome("a", Status.FAILED))

    entry = ledger.record(Outcome("a", Status.FAILED))

    assert not entry.is_settled(max_attempts=3)
    assert entry.is_settled(max_attempts=2)


def test_truncated_last_line_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "shard.jsonl"
    Ledger(path).record(Outcome("a", Status.DONE))
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"key": "b", "sta')

    assert set(Ledger(path).latest()) == {"a"}


def test_ledgers_of_all_shards_are_merged(tmp_path: Path) -> None:
    Ledger.for_shard(tmp_path, Shard(0, 2)).record(Outcome("a", Status.DONE))
    Ledger.for_shard(tmp_path, Shard(1, 2)).record(Outcome("b", Status.UNAVAILABLE))

    merged = read_ledgers(tmp_path)

    assert {key: entry.status for key, entry in merged.items()} == {
        "a": Status.DONE,
        "b": Status.UNAVAILABLE,
    }


def test_attempts_carry_over_when_the_shard_count_changes(tmp_path: Path) -> None:
    Ledger.for_shard(tmp_path, Shard.whole()).record(Outcome("a", Status.FAILED))

    entry = Ledger.for_shard(tmp_path, Shard(0, 4)).record(Outcome("a", Status.FAILED))

    assert entry.attempts == 2
