import jsonlines
import random
from tqdm import tqdm
import os

raw_path = '../../kilt_knowledgeinput.json'
random_seed = 42
target_dir = './datasets/'


def split_wikipedia_with_index():
    random.seed(random_seed)
    with jsonlines.open(raw_path, 'r') as reader:
        data = [i['wikipedia_id'] for i in tqdm(reader)]
    random.shuffle(data)
    with jsonlines.open(target_dir + 'D0-id.jsonl', 'w') as writer:
        writer.write_all(data[: int(len(data) * 0.6)])
    with jsonlines.open(target_dir + 'D1-id.jsonl', 'w') as writer:
        writer.write_all(data[int(len(data) * 0.6): int(len(data) * 0.7)])
    with jsonlines.open(target_dir + 'D2-id.jsonl', 'w') as writer:
        writer.write_all(data[int(len(data) * 0.7): int(len(data) * 0.8)])
    with jsonlines.open(target_dir + 'D3-id.jsonl', 'w') as writer:
        writer.write_all(data[int(len(data) * 0.8): int(len(data) * 0.9)])
    with jsonlines.open(target_dir + 'D4-id.jsonl', 'w') as writer:
        writer.write_all(data[int(len(data) * 0.9):])

def write_index_into_item():
    index_dict = {}
    for i in range(5):
        index_dict[i] = set()
        index_path = target_dir + 'D' + str(i) + '-id.jsonl'
        with jsonlines.open(index_path) as reader:
            for item in tqdm(reader):
                index_dict[i].add(item)

    writer_dict = {}
    for i in range(5):
        target_path = target_dir + 'D' + str(i) + '.jsonl'
        writer_dict[i] = jsonlines.open(target_path, 'w')

    with jsonlines.open(raw_path, 'r') as reader:
        for item in tqdm(reader):
            for i in range(5):
                if item['wikipedia_id'] in index_dict[i]:
                    writer_dict[i].write(item)
    
    for i in range(5):
        writer_dict[i].close()


def split_kilt(input_dir, output_dir, prefix):
    index_dict = {}
    for i in range(5):
        index_dict[i] = set()
        index_path = target_dir + 'D' + str(i) + '-id.jsonl'
        with jsonlines.open(index_path) as reader:
            for item in tqdm(reader):
                index_dict[i].add(item)

    input_files = os.listdir(input_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        with jsonlines.open(input_path) as reader:
            for item in tqdm(reader, desc=input_file):
                flag = 0
                for ans in item['output']:
                    if "provenance" in ans.keys():
                        for prov in ans["provenance"]:
                            for i in range(5):
                                if prov['wikipedia_id'] in index_dict[i]:
                                    flag = max(flag, i)
                tmp_dir = os.path.join(output_dir, input_file.rstrip('.jsonl'))
                if not os.path.exists(tmp_dir):
                    os.mkdir(tmp_dir)
                output_file = os.path.join(tmp_dir, prefix + str(flag) + '.jsonl')
                with jsonlines.open(output_file, 'a') as writer:
                    writer.write(item)
    

if __name__ == "__main__":
    # split_wikipedia_with_index()
    # write_index_into_item()
    # split_kilt('datasets/kilt-dev/', 'datasets/kilt++-test', 'Q')
    #split_kilt('datasets/kilt-train/', 'datasets/kilt++-train', 'R')