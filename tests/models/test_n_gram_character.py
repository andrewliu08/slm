from collections import Counter

import torch

from models.n_gram_character import NGramCharacter


def test_one_gram_gram_and_next_char():
    model = NGramCharacter(n=1)
    words = ["a", "ac", "babc"]

    actual_grams, actual_chars = model.gram_and_next_char(words)
    expected_grams = ["$", "a", "$", "a", "c", "$", "b", "a", "b", "c"]
    expected_chars = ["a", "$", "a", "c", "$", "b", "a", "b", "c", "$"]
    assert actual_grams == expected_grams
    assert expected_chars == actual_chars


def test_three_gram_gram_and_next_char():
    model = NGramCharacter(n=3)
    words = ["a", "ac", "babc"]

    actual_grams, actual_chars = model.gram_and_next_char(words)
    expected_grams = ["$", "$a", "$", "$a", "$ac", "$", "$b", "$ba", "bab", "abc"]
    expected_chars = ["a", "$", "a", "c", "$", "b", "a", "b", "c", "$"]
    assert actual_grams == expected_grams
    assert expected_chars == actual_chars


def test_fit():
    model = NGramCharacter(n=1)
    words = ["a", "ac", "babc"]
    model.fit(words)

    assert model.grams == ["$", "a", "b", "c"]
    assert model.chars == ["$", "a", "b", "c"]

    expected_probs = torch.tensor(
        [
            [0.0, 0.6666, 0.3333, 0.0],
            [0.3333, 0.0, 0.3333, 0.3333],
            [0.0, 0.5, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(model.probs, expected_probs, atol=1e-4)


def test_generate_character():
    model = NGramCharacter(n=1)
    model.grams = ["$", "a", "b"]
    model.chars = ["$", "a", "b"]
    model.gram_to_idx = {"$": 0, "a": 1, "b": 1}
    model.char_to_idx = {"$": 0, "a": 1, "b": 1}
    model.probs = torch.tensor(
        [
            [0.0, 0.75, 0.25],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    generator = torch.Generator().manual_seed(12345)
    results = [model.generate_character(generator, "$") for _ in range(100)]
    actual_freq = Counter(results)
    # "a" frequency should be close to 75 and "b" close to 25
    expected_freq = Counter({"a": 78, "b": 22})

    assert actual_freq == expected_freq


def test_generate_word():
    model = NGramCharacter(n=1)
    model.grams = ["$", "a", "b"]
    model.chars = ["$", "a", "b"]
    model.gram_to_idx = {"$": 0, "a": 1, "b": 1}
    model.char_to_idx = {"$": 0, "a": 1, "b": 1}
    model.probs = torch.tensor(
        [
            [0.0, 0.5, 0.5],
            [0.9, 0.0, 0.1],
            [0.9, 0.0, 0.1],
        ]
    )

    generator = torch.Generator().manual_seed(12345)
    results = [model.generate_word(generator) for _ in range(100)]
    actual_freq = Counter(results)
    expected_freq = Counter({"a": 49, "b": 46, "ab": 2, "bb": 2, "abb": 1})

    assert actual_freq == expected_freq
