import json

class KhmerVocab:
    def __init__(self, charset_path):
        self.blank_token = "<blank>"

        with open(charset_path) as f:
            charset = json.load(f)
        characters = "".join(charset.values())

        self.idx_to_char = [self.blank_token] + list(characters)

        self.char_to_idx = {
            char: idx for idx, char in enumerate(self.idx_to_char)
        }

    def encode(self, text):
        encoded = []

        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                print(f"Unknown character skipped: {char}")

        return encoded

    def decode(self, indices):
        text = ""
        previous = None

        for idx in indices:
            if idx != 0 and idx != previous:
                text += self.idx_to_char[idx]

            previous = idx

        return text

    def __len__(self):
        return len(self.idx_to_char)
