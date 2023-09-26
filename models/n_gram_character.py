from typing import Dict, List, Tuple

import torch


class NGramCharacter:
    """
    Predicts the next character based on the frequency character
    x follows the previous n characters in the training set.
    """

    def __init__(self, n: int):
        self.n = n
        self.grams: List[str] = []
        self.chars: List[str] = []
        self.gram_to_idx: Dict[str, int] = {}
        self.char_to_idx: Dict[str, int] = {}
        self.probs: torch.tensor = torch.zeros(0)

        self.start_char = "$"
        self.end_char = "$"

    def get_gram(self, word: str, idx: int) -> str:
        return word[max(idx - self.n, 0) : idx]

    def gram_and_next_char(self, words: List[str]) -> Tuple[List[str], List[str]]:
        grams, next_char = [], []

        for word in words:
            word = self.start_char + word + self.end_char
            for i in range(1, len(word)):
                grams.append(self.get_gram(word, i))
                next_char.append(word[i])

        return grams, next_char

    def fit(self, words: List[str]) -> None:
        grams, next_char = self.gram_and_next_char(words)
        self.grams = sorted(set(grams))
        self.chars = sorted(set(next_char))
        self.gram_to_idx = {gram: idx for idx, gram in enumerate(self.grams)}
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}

        # Count frequencies
        self.probs = torch.zeros(len(self.grams), len(self.chars))
        for g, c in zip(grams, next_char):
            gram_idx = self.gram_to_idx[g]
            char_idx = self.char_to_idx[c]
            self.probs[gram_idx, char_idx] += 1.0

        # Normalize frequencies to probabilities
        self.probs /= self.probs.sum(dim=1, keepdim=True)

    def generate_character(self, generator: torch.Generator, prev_gram: str) -> str:
        gram_idx = self.gram_to_idx[prev_gram]
        char_idx = torch.multinomial(
            self.probs[gram_idx], num_samples=1, replacement=True, generator=generator
        ).item()
        return self.chars[char_idx]

    def generate_word(self, generator: torch.Generator) -> str:
        word = self.start_char
        while len(word) == 1 or word[-1] != self.end_char:
            prev_gram = self.get_gram(word, len(word))
            word += self.generate_character(generator, prev_gram)

        # Remove self.start_char and self.end_char from the word
        return word[1:-1]
