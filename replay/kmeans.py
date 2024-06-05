import jsonlines
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pickle
import torch
import faiss
import numpy as np
from tqdm import tqdm
import random


def enocde_documents(prefile, save_file, total, batch_size=1024):
    tokenizer = BertTokenizer.from_pretrained('../pretrain_tasks/sf/bert-base-uncased/')
    model = BertModel.from_pretrained('../pretrain_tasks/sf/bert-base-uncased/').eval().to('cuda:0')
    num_batch = total // batch_size if total % batch_size == 0 else total // batch_size + 1
    all_ids, all_tensors = [], []
    with torch.no_grad():
        with jsonlines.open(prefile) as reader:
            for i in tqdm(range(num_batch)):
                batch = None
                if i == num_batch - 1 and total % batch_size != 0:
                    batch = [reader.read() for j in range(total % batch_size)]
                else:
                    batch = [reader.read() for j in range(batch_size)]
                text = [j['wikipedia_title'] for j in batch]
                ids = [int(j['wikipedia_id']) for j in batch]
                encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, add_special_tokens=True, padding=True).to('cuda:0')
                output = model(**encoded_input).last_hidden_state[:, 0, :].cpu().numpy()
                all_tensors.append(output)
                all_ids.extend(ids)
        all_tensors = np.concatenate(all_tensors, axis=0)
        print(all_tensors.shape)
        with open(save_file + '.tensor', 'wb') as f:
            pickle.dump(all_tensors, f)
        
        with open(save_file + '.id', 'wb') as f:
            pickle.dump(all_ids, f)


def cluster(x, ncentroids=1024, niter=20, verbose=True):
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
    kmeans.train(x)
    return kmeans

def load_tensor_and_id2wiki(path):
    with open(path + '.tensor', 'rb') as f:
        concat_tensor = pickle.load(f)

    with open(path + '.id', 'rb') as f:
        ids = pickle.load(f)
    
    idx2wiki = {idx: wiki for idx, wiki in enumerate(ids)}

    return concat_tensor, idx2wiki


def  query_with_kmeans(kmeans, ncentroids, idx2wiki, concat_tensor):
    D, I = kmeans.index.search(concat_tensor, 1)
    I = I.reshape(-1)

    cent2wiki = {i: [] for i in range(ncentroids)}
    for idx, cent in enumerate(I):
        cent2wiki[cent].append(idx2wiki[idx])
    return cent2wiki

def write_index_into_item(session):
    raw_path = '../../../kilt_knowledgesource.json'
    re_set = set()

    index_path = 're-D' + str(session) + '-id.jsonl'
    with jsonlines.open(index_path) as reader:
        for item in tqdm(reader):
            re_set.add(item)

    target_path = 're-D' + str(session) + '.jsonl'
    with jsonlines.open(target_path, 'w') as writer:
        with jsonlines.open(raw_path, 'r') as reader:
            for item in tqdm(reader):
                    if item['wikipedia_id'] in re_set:
                        writer.write(item)

if_load = True

if __name__ == "__main__":
    prefile = '../data/filtered_D0.jsonl'
    save_file = './semantic-D0'
    # enocde_documents(prefile, save_file, total=301601)    # D1
    # enocde_documents(prefile, save_file, total=1000)      # D1-dev
    # enocde_documents(prefile, save_file, total=1807697)     # D0
    
    mem_file = './semantic-D0'
    query_file = './semantic-D1'
    ncentroids = 10000

    if if_load:
        with open(mem_file + '.cluster', 'rb') as f:
            mem_cent2wiki = pickle.load(f)
        with open(query_file + '.cluster', 'rb') as f:
            query_cent2wiki = pickle.load(f)

        session = 2
        test = mem_cent2wiki[1] + query_cent2wiki[1]
        test = [str(i) for i in test]
        with jsonlines.open('re-D' + str(session) + '-id.jsonl', 'w') as writer:
            writer.write_all(test)
        write_index_into_item(session)
        exit()

        session = 1
        index_path = 're-D' + str(session) + '-id.jsonl'
        with jsonlines.open(index_path, 'w') as writer:
            for i in range(ncentroids):
                if len(query_cent2wiki[i]) > 0:
                    num_samples = min(len(mem_cent2wiki[i]), len(query_cent2wiki[i]))
                    samples = random.sample(mem_cent2wiki[i], k=num_samples)
                    samples = [str(sample) for sample in samples]
                    writer.write_all(samples)

        write_index_into_item(session)

    else:
        concat_tensor, idx2wiki = load_tensor_and_id2wiki(mem_file)
        kmeans = cluster(concat_tensor, ncentroids)

        mem_cent2wiki = query_with_kmeans(kmeans, ncentroids, idx2wiki, concat_tensor)
        with open(mem_file + '.cluster', 'wb') as f:
            pickle.dump(mem_cent2wiki, f)
   
        query_tensor, query_idx2wiki = load_tensor_and_id2wiki(query_file)
        query_cent2wiki = query_with_kmeans(kmeans, ncentroids, query_idx2wiki, query_tensor)
        with open(query_file + '.cluster', 'wb') as f:
            pickle.dump(query_cent2wiki, f)
    
        print(mem_cent2wiki)
        print(query_cent2wiki)