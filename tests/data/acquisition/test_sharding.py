import pytest

from signworld.acquisition.sharding import Shard

KEYS = [f"video-{index}" for index in range(1000)]


def test_shards_partition_the_keys() -> None:
    selections = [set(Shard(index, 8).select(KEYS)) for index in range(8)]

    assert set().union(*selections) == set(KEYS)
    assert sum(len(selection) for selection in selections) == len(KEYS)


def test_membership_does_not_depend_on_key_order() -> None:
    shard = Shard(3, 8)

    assert set(shard.select(KEYS)) == set(shard.select(reversed(KEYS)))


@pytest.mark.parametrize(("index", "count"), [(-1, 4), (4, 4), (0, 0)])
def test_invalid_shards_are_rejected(index: int, count: int) -> None:
    with pytest.raises(ValueError, match="shard"):
        Shard(index, count)
