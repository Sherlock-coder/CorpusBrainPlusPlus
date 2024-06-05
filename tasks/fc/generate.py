import jsonlines
from tqdm import tqdm
import argparse
import random
from tasks import *

source = 'all_generated_data.jsonl8.session4'
target = 'all_generated_data.jsonl5.session4'
import random

def split_and_select(input_string, n):
    # 分割字符串为单词列表
    words = input_string.split()
    
    if n > len(words):
        return "Error: n is larger than the number of words"
    
    # 随机选择起始索引
    start_index = random.randint(0, len(words) - n)
    
    # 从起始索引开始选取n个单词
    selected_words = words[start_index:start_index + n]
    
    # 还原为字符串
    selected_string = ' '.join(selected_words)
    
    return selected_string



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
    source, target = args.input, args.output

    num_pseudo_queries = 3


    with jsonlines.open(source) as reader:
        with jsonlines.open(target, 'w') as writer:
            for item in tqdm(reader):
                pairs = iss(item, 3)
                for pair in pairs:
                    new = {}
                    for pair in pairs:
                        new['source'] = item['wikipedia_title'] +  ' ' + split_and_select(pair['source'], 10)
                        new['target'] = pair['target']
                        writer.write(new)