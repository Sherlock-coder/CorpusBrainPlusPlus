import json
from sys import argv
from tqdm import tqdm
import jsonlines


if __name__ == "__main__":
    save_file = jsonlines.open(f'./data/filtered_{argv[1]}', 'w')
    with jsonlines.open(f'./datasets/{argv[1]}', 'r') as reader:
        for data in tqdm(reader):
            title = data['wikipedia_title']
            if len(title) <= 3:
                continue
            content = []
            for sentence in data['text']:
                if '::::' not in sentence:
                    content.append(sentence)
            all_content = ' '.join(content)
            len_content = len(all_content.split(' '))
            if len_content < 128:
                continue
            save_file.write(data)
    save_file.close()
