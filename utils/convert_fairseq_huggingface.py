import os
import sys

import torch
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    FlaxBartForConditionalGeneration,
    TFBartForConditionalGeneration,
)

from genre.fairseq_model import GENRE
import argparse

def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = torch.nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer




parser = argparse.ArgumentParser()
parser.add_argument(
    "input",
    type=str,
    default="",
    help="input file",
)
args = parser.parse_args()
fairseq_path = args.input
hf_path = 'hf-' + args.input


fairseq_model = GENRE.from_pretrained(fairseq_path, checkpoint_file='checkpoint_best.pt').eval()
config = BartConfig(vocab_size=50264)
hf_model = BartForConditionalGeneration(config).eval()
hf_tokenizer = BartTokenizer.from_pretrained("./bart-large")
hf_tokenizer.save_pretrained(hf_path)

# Convert pytorch

state_dict = fairseq_model.model.state_dict()
remove_ignore_keys_(state_dict)
state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
hf_model.model.load_state_dict(state_dict, strict=True)
hf_model.lm_head = make_linear_from_emb(hf_model.model.shared)
hf_model.save_pretrained(hf_path)
