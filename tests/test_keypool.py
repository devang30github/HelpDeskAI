import pytest

from app.core.config import Settings
from app.core.keypool import KeyPool


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t


def test_keys_are_cleaned_and_rotated():
    c = Clock()
    pool = KeyPool([" a ", "b", "a", ""], clock=c.now)
    assert len(pool) == 2
    assert [pool.pick()[0] for _ in range(4)] == ["a", "b", "a", "b"]


def test_cooldown_skips_a_key_until_it_expires():
    c = Clock()
    pool = KeyPool(["a", "b"], clock=c.now)
    pool.cooldown("a", 10)
    assert [pool.pick()[0] for _ in range(2)] == ["b", "b"]
    c.t = 11
    assert pool.pick()[0] == "a"


def test_all_cooling_returns_the_earliest_with_its_wait():
    c = Clock()
    pool = KeyPool(["a", "b"], clock=c.now)
    pool.cooldown("a", 10)
    pool.cooldown("b", 5)
    assert pool.pick() == ("b", 5.0)


def test_disabled_keys_are_never_used_and_exhaustion_raises():
    pool = KeyPool(["a", "b"])
    pool.disable("a")
    assert [pool.pick()[0] for _ in range(3)] == ["b", "b", "b"]
    pool.disable("b")
    with pytest.raises(RuntimeError):
        pool.pick()


def test_labels_never_contain_the_key():
    pool = KeyPool(["secret-key-1", "secret-key-2"])
    assert pool.label("secret-key-2") == "key-2/2"


def test_settings_merge_and_deduplicate_keys():
    s = Settings(groq_api_key="a", groq_api_keys="b, c\nd,a")
    assert s.groq_key_list() == ["b", "c", "d", "a"]
    assert Settings(groq_api_key="", groq_api_keys="").groq_key_list() == []