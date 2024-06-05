import pickle

import jsonlines
from tqdm import tqdm
from genre.trie import Trie
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./bart-large')
tokenized_titles = []
path = 'datasets/'

for index in range(5):
    with jsonlines.open(path + 'D' + str(index) +'.jsonl') as items:
        for item in tqdm(items, desc='D' + str(index)):
            title = item['wikipedia_title']
            tokenized_titles.append([2] + tokenizer(title, add_special_tokens=False)['input_ids'] + [2])

    constructed_trie = Trie(tokenized_titles)
    with open('D0-' + str(index) + '.pkl', 'wb') as f:
        pickle.dump(constructed_trie.trie_dict, f)
    print(len(constructed_trie))

print("Successfully construct new tries.")