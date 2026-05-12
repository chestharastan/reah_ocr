import json


class KhmerVocab:
    """
    Vocabulary for the seq2seq chunkmerge model.

    Reserved ids:
        0 -> <pad>
        1 -> <bos>
        2 -> <eos>
        3 -> <unk>
    Character ids start at 4.
    """

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, charset_path):
        with open(charset_path) as f:
            charset = json.load(f)
        characters = "".join(charset.values())

        self.idx_to_char = [self.PAD, self.BOS, self.EOS, self.UNK] + list(characters)
        self.char_to_idx = {c: i for i, c in enumerate(self.idx_to_char)}

        self.pad_id = self.char_to_idx[self.PAD]
        self.bos_id = self.char_to_idx[self.BOS]
        self.eos_id = self.char_to_idx[self.EOS]
        self.unk_id = self.char_to_idx[self.UNK]

    def encode(self, text):
        return [self.char_to_idx.get(c, self.unk_id) for c in text]

    def decode(self, indices):
        """Convert a list of ids back to text, stopping at the first EOS and
        skipping PAD/BOS/UNK so the output is suitable for CER comparison."""
        skip = {self.pad_id, self.bos_id, self.unk_id}
        chars = []
        for idx in indices:
            idx = int(idx)
            if idx == self.eos_id:
                break
            if idx in skip:
                continue
            chars.append(self.idx_to_char[idx])
        return "".join(chars)

    def __len__(self):
        return len(self.idx_to_char)
