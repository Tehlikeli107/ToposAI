from typing import Dict, List


class TopologicalTokenizer:
    """
    Small tokenizer that greedily merges high-implication adjacent pairs.

    Pair strength is estimated as `count(A, B) / count(A)`. This is a simple
    morphism-inspired tokenizer baseline, not a replacement for production BPE
    tokenizers without benchmark evidence.
    """

    def __init__(self, vocab_size=1000):
        self.target_vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.merges: Dict[tuple[str, str], str] = {}
        self.reverse_vocab: Dict[int, str] = {}

    def _compute_topological_morphisms(self, token_list):
        pair_counts = {}
        single_counts = {}

        for i in range(len(token_list) - 1):
            A = token_list[i]
            B = token_list[i + 1]

            single_counts[A] = single_counts.get(A, 0) + 1
            pair_counts[(A, B)] = pair_counts.get((A, B), 0) + 1

        morphism_strengths = {}
        for (A, B), count in pair_counts.items():
            morphism_strengths[(A, B)] = count / single_counts[A]

        return morphism_strengths

    def train(self, text: str):
        print(f"\n[TOPOLOGICAL TOKENIZER] training target vocab={self.target_vocab_size}")

        unique_chars = sorted(set(text))
        self.vocab = {c: i for i, c in enumerate(unique_chars)}
        self.reverse_vocab = {i: c for i, c in enumerate(unique_chars)}

        current_tokens = [c for c in text]
        current_id = len(self.vocab)

        merge_count = 0
        while len(self.vocab) < self.target_vocab_size:
            morphisms = self._compute_topological_morphisms(current_tokens)
            if not morphisms:
                break

            best_pair = max(morphisms.items(), key=lambda x: (x[1], x[0]))[0]
            best_strength = morphisms[best_pair]
            if best_strength < 0.01:
                break

            new_token_str = best_pair[0] + best_pair[1]
            if new_token_str in self.vocab:
                break

            self.vocab[new_token_str] = current_id
            self.reverse_vocab[current_id] = new_token_str
            self.merges[best_pair] = new_token_str

            new_tokens = []
            i = 0
            while i < len(current_tokens):
                if i < len(current_tokens) - 1 and (current_tokens[i], current_tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token_str)
                    i += 2
                else:
                    new_tokens.append(current_tokens[i])
                    i += 1

            current_tokens = new_tokens
            current_id += 1
            merge_count += 1

            if merge_count % 100 == 0:
                print(f"  > {merge_count} merges, vocab={len(self.vocab)}")

        print(f"[TOPOLOGICAL TOKENIZER] training complete, vocab={len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """Convert text to token ids."""
        tokens = [c for c in text]

        for (A, B), new_token_str in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == A and tokens[i + 1] == B:
                    new_tokens.append(new_token_str)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self.vocab.get(t, self.vocab.get(" ", 0)) for t in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token ids back to text."""
        return "".join(self.reverse_vocab.get(t, "") for t in token_ids)

    def save(self, filepath: str):
        """Save tokenizer state as JSON."""
        import json

        merges_str_keys = {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()}
        data = {
            "vocab_size": self.target_vocab_size,
            "vocab": self.vocab,
            "merges": merges_str_keys,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """Load tokenizer state from JSON."""
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.target_vocab_size = data["vocab_size"]
        self.vocab = data["vocab"]
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        self.merges = {}
        for k, v in data["merges"].items():
            parts = k.split("|")
            if len(parts) == 2:
                self.merges[(parts[0], parts[1])] = v
