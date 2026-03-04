from tokenizers import ByteLevelBPETokenizer
import os

class BPETokenizer:
    def __init__(self, vocab_size: int, min_frequency: int = 2):
        self.tokenizer = ByteLevelBPETokenizer()
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def train(self, files):
        self.tokenizer.train(files=files, vocab_size = self.vocab_size,
                             min_frequency=self.min_frequency,
                             special_tokens=['<pad>', '<s>', '</s>', '<unk>', '<mask>'])
    
    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.tokenizer.save_model(directory)

    def load(self, directory):
        self.tokenizer = ByteLevelBPETokenizer(
            os.path.join(directory, "vocab.json"),
            os.path.join(directory, "merges.txt")
        )