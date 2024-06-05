from tasks import *
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import argparse

prefile = '../../data/filtered_D2.jsonl'
# prefile = '../../replay/re-D1.jsonl'
generate_file = 'generated_data.jsonl.session2'
num_samples = 3
batch_size = 128 * 4     # real batch size ~= batch_size * num_samples
num_labels = 154


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
    print(args)

    prefile = args.input
    generate_file = args.output

    with jsonlines.open(prefile, 'r') as reader:
        data = [i for i in reader]
    total = len(data)

    tokenizer = AutoTokenizer.from_pretrained('./pretrain_tasks/sf/bert-finetune/')
    model = AutoModelForSequenceClassification.from_pretrained('./pretrain_tasks/sf/bert-finetune/', num_labels=num_labels).eval().to('cuda:0')

    with jsonlines.open('./pretrain_tasks/sf/rel2id.jsonl') as reader:
        rel2id = reader.read()
    id2rel = {v: k for k, v in rel2id.items()}

    with torch.no_grad():
        with jsonlines.open(generate_file, 'w') as writer:
            num_batch = total // batch_size if total % batch_size == 0 else total // batch_size + 1
            for i in tqdm(range(num_batch)):
                batch = data[batch_size * i: batch_size * (i + 1)]
                texts, titles = [], []
                for entry in batch:
                    pairs = iss(entry, num_samples)
                    texts.extend([j['source'] for j in pairs])
                    titles.extend([j['target'].split(' |')[0] for j in pairs])
                encoded_input = tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, add_special_tokens=True, padding=True).to('cuda:0')
                output = model(**encoded_input)
                predictions = torch.argmax(output[0], dim=1)
                for sample, title, prediction in zip(texts, titles, predictions):
                    writer.write({'source': title + ' [SEP] ' + id2rel[prediction.item()], 'target': title, 'sample': sample})
                