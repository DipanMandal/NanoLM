class NanoLMConfig():
    def __init__(self):
        self.d_model = 16
        self.sequence_length = 100
        self.dropout = 0.2
        self.num_heads = 1
        self.d_hidden = 8
        self.vocab_size = 500
        self.n_encoder_layers = 2
        self.use_bias = True