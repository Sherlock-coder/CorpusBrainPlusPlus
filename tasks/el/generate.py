import sys
sys.path.append('..')

from tasks import *
from tqdm import tqdm
import random
import pickle
from genre.trie import Trie
import argparse

file = '../../data/filtered_D1.jsonl'
# file = '../../replay/re-D4.jsonl'
generate_file = 'generated_data.jsonl7.session1'
# 7 is the code for hip_all_anchor
# 9 is the code fro hip_all_anchor + hip_modify_1 (num_samples=1)
num_samples = 1

all_titles = []


def hip_all_anchor(item):
    text = item['text']
    ret = []
    if "anchors" in item.keys():
        for anchor in item['anchors']:
            if "wikipedia_title" in anchor.keys():
                pid = anchor['paragraph_id']
                start, end = anchor["start"], anchor['end']
                target = anchor['wikipedia_title']
                source = text[pid][: start] + '[START_ENT] ' + text[pid][start: end] + ' [END_ENT]' + text[pid][end:]
                ret.append({'source': source, 'target': target})
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=str,
        default="",
        help="input file",
    )
    parser.add_argument(
        "output",
        type=str,
        default="",
        help="output file",
    )
    args = parser.parse_args()
    file = args.input
    generate_file = args.output
    print(file, generate_file)

    with jsonlines.open(file, 'r') as reader:
        data = [i for i in reader]
        
    for item in data:
        all_titles.append(item['wikipedia_title'])

    with jsonlines.open(generate_file, 'w') as writer:
        for item in tqdm(data):
            writer.write_all(hip_all_anchor(item))
