FTransformerDecoderConfig = {
    "Small": {
        "nhead": 4,
        "num_layers": 6,
        "dim": 192,
    },
    "Base": {
        "nhead": 8,
        "num_layers": 6,
        "dim": 256,
    },
    "Large": {
        "nhead": 8,
        "num_layers": 8,
        "dim": 320,
    }
}

TextEncoderConfig= {
    "Small": {
        "nhead": 4,
        "num_layers": 3,
        "dim": 192,
    },
    "Base": {
        "nhead": 4,
        "num_layers": 3,
        "dim": 256,
    },
    "Large": {
        "nhead": 8,
        "num_layers": 4,
        "dim": 320,
    }
}
PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2
MAX_SEQ_LENGTH = 160
