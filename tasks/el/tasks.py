import json
import random
from sys import argv

from tqdm import tqdm
import jsonlines

def get_anchors(anchors):
    '''Get anchors from text.'''
    numbers = [0, 1, 2, 3, 4]
    anchor_number = random.choices(numbers, weights=(70, 20, 5, 3, 2), k=1)[0]
    min_k = min(anchor_number, len(anchors))
    anchor_sets = random.sample(anchors, k=min_k)
    anchor_sets = [
        anchor['wikipedia_title'] for anchor in anchor_sets
        if 'wikipedia_title' in anchor
    ]
    anchor_sets = list(set(anchor_sets))
    return anchor_sets

def lps(item, n):
    text = item['text']
    title = item['wikipedia_title']
    anchors = item['anchors']
    content = [text[0]]
    for s in text[1:]:      # different from the original implementation of Chen, we believe it's more reasonable
        if '::::' in s:
            continue
        len_s = len(s.split(' '))
        if len_s > 10:
            content.append(s)

    sentence_sets = []
    paragraph = None
    if  n >= 1:
        paragraph = ' '.join([s.strip() for s in content[1:4]])
        sentence_sets.append(paragraph)
    if len(content) > 6 and n >= 2:
        paragraph = ' '.join([s.strip() for s in content[4:7]])
        sentence_sets.append(paragraph)
    if len(content) > 9 and n >= 3:
        paragraph = ' '.join([s.strip() for s in content[7:10]])
        sentence_sets.append(paragraph)
    sentence_sets = list(set(sentence_sets))
    sentence_sets = [sentence.strip() for sentence in sentence_sets]
    ret = []
    for sentence in sentence_sets:
        anchor_sets = get_anchors(anchors)
        target = title
        if anchor_sets != []:
            target = title + ' |' + ' |'.join(anchor_sets)
        ret.append({'source': sentence, 'target': target})
    return ret

def iss(item, n):
    text = item['text']
    title = item['wikipedia_title']
    anchors = item['anchors']
    content = [text[0]]
    for s in text[1:]:  # different from the original implementation of Chen, we believe it's more reasonable
        if '::::' in s:
            continue
        len_s = len(s.split(' '))
        if len_s > 10:
            content.append(s)

    sentence_sets = []
    if len(content) > 1:
        sentence_sets.append(content[1])
    random_sentences = []
    if len(content) > 2 and n >= 2:
        min_k = min(n - 1, len(content) - 2)
        random_sentences = random.sample(content[2:], k=min_k)
    sentence_sets.extend(random_sentences)
    sentence_sets = list(set(sentence_sets))
    sentence_sets = [sentence.strip() for sentence in sentence_sets]
    ret = []
    for sentence in sentence_sets:
        anchor_sets = get_anchors(anchors)
        target = title
        if anchor_sets != []:
            target = title + ' |' + ' |'.join(anchor_sets)
        ret.append({'source': sentence, 'target': target})
    return ret

def hip(item, n):   # need to be improved in terms of efficiency, and note: tend to be empty!
    text = item['text']
    anchors = item['anchors']
    not_valid_anchor = True
    current_valid_anchor = 0
    all_paragraph_len = len(text)
    added_anchor_list = []
    ret = []
    while current_valid_anchor < 3 and len(anchors) > 0 and all_paragraph_len > 3:
        selected_anchor = random.choice(anchors)
        paragraph_id = selected_anchor['paragraph_id']
        if paragraph_id < 2 or \
                paragraph_id == all_paragraph_len or \
                'wikipedia_title' not in selected_anchor or \
                '::::' in text[paragraph_id]:
            anchors.remove(selected_anchor)
            continue
        if (selected_anchor['wikipedia_title'], paragraph_id) in added_anchor_list:
            anchors.remove(selected_anchor)
            continue
        left_id = paragraph_id - 1
        while '::::' in text[left_id] and left_id > 1:
            left_id -= 1
        if left_id == 1:
            anchors.remove(selected_anchor)
            continue
        right_id = paragraph_id + 1
        while right_id < all_paragraph_len and '::::' in text[right_id]:
            right_id += 1
        if right_id == all_paragraph_len:
            anchors.remove(selected_anchor)
            continue
        target_text = [text[left_id], text[paragraph_id], text[right_id]]
        target_text = ' '.join([s.strip() for s in target_text])
        if target_text == '':
            continue
        target = selected_anchor['wikipedia_title']
        ret.append({'source': target_text, 'target': target})
        added_anchor_list.append((target, paragraph_id))
        current_valid_anchor += 1
    return ret[: n]

if __name__ == '__main__':
    with jsonlines.open('./filtered_D1.jsonl.dev') as reader:
        for item in reader:
            print(item)
            print(iss(item, 3))
            print(lps(item, 3))
            print(hip(item, 3))
