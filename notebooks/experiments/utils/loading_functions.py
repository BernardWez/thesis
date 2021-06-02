import copy
import re
import xml.etree.ElementTree as et

import numpy as np


def load_file(file_path):
    """
    Loads file and returns all the articles
    """
    # Load the data
    tree = et.parse(file_path)
    root = tree.getroot()

    return root


def process_article(article, filtered, file_path):
    """
    Takes article and process into desired structure
    """
    if 'GeoWebNews' in file_path:
        if filtered:
            return {'text': article.find('text').text.replace('\n', ' '),
                    'entities': sorted([{'text': top.find('extractedName').text,
                                         'start_pos': int(top.find('start').text),
                                         'end_pos': int(top.find('end').text)} for top in article.findall('toponyms/toponym')
                                        if top.find('latitude') != None and top.find('longitude') != None], key=lambda k: k['start_pos'])}

        else:
            return {'text': article.find('text').text.replace('\n', ' '),
                    'entities': sorted([{'text': top.find('extractedName').text,
                                         'start_pos': int(top.find('start').text),
                                         'end_pos': int(top.find('end').text)} for top in article.findall('toponyms/toponym')], key=lambda k: k['start_pos'])}

    elif not filtered:
        return {'text': article.find('text').text.replace('\n', ' '),
                'entities': sorted([{'text': top.find('phrase').text,
                                     'start_pos': int(top.find('start').text),
                                     'end_pos': int(top.find('end').text)} for top in article.findall('toponyms/toponym')
                                    ], key=lambda k: k['start_pos'])}

    else:
        return {'text': article.find('text').text.replace('\n', ' '),
                'entities': sorted([{'text': top.find('phrase').text,
                                     'start_pos': int(top.find('start').text),
                                     'end_pos': int(top.find('end').text)} for top in article.findall('toponyms/toponym')
                                    if top.find('gaztag/lat') != None and top.find('gaztag/lon') != None
                                    ], key=lambda k: k['start_pos'])}


def process_articles(root, filtered, file_path):
    """
    Takes articles and processes them into desired structure
    """
    data = []

    for article in root:

        data.append(process_article(article, filtered, file_path))

    return data


def prepare_data(file_path, filtered, split, word_limit=2000):

    root = load_file(file_path)

    data = process_articles(root, filtered, file_path)

    if split:

        data = [split for splits in [split_article(
            entry, word_limit) for entry in data] for split in splits]

    # Clean white spaces
    data = [clean_whitespaces(i) for i in data]

    # Align annotations
    data = [align_annotations(i) for i in data]

    return data


def split_article(entry, word_limit):

    # Get all punctuation indices
    punc_indices = set([m.start() for m in re.finditer('\.', entry['text'])])

    # Get all indices belonging to toponyms
    toponym_indices = set()

    for entity in entry['entities']:
        for i in range(entity['start_pos'], entity['end_pos']+1):
            toponym_indices.add(i)

    # Only keep punctuation indices not belonging to toponyms
    punc_indices_filtered = sorted(
        list(punc_indices.difference(toponym_indices)))

    # Split article in sentences
    s = entry['text']
    split_article = []
    position = 0

    # stuff is still cool

    for idx in punc_indices_filtered:

        sub_text = s[position:idx+1]
        len_sub_text = len(sub_text)

        split_article.append(sub_text)
        position += len_sub_text

    split_article.append(s[position:])

    # Merge article in sub components
    sub_components = []
    component = ''

    for sentence in split_article:
        if len(sentence) + len(component) < word_limit:
            component += sentence
        else:
            sub_components.append(component)
            component = sentence

    sub_components.append(component)

    # Update the entities annotations (i.e. correct the start/end positions)
    tmp = [0] + list(np.cumsum([len(i) for i in sub_components]))
    tmp

    split_entities = [[entity for entity in entry['entities']
                       if entity['end_pos'] <= curr and entity['start_pos'] >= prev]
                      for prev, curr in zip(tmp[:-1], tmp[1:])]

    updated_entities = [[{'text': top['text'],
                          'start_pos': top['start_pos'] - index_shift,
                          'end_pos': top['end_pos'] - index_shift} for top in entity]
                        for entity, index_shift in zip(split_entities[:], tmp[:-1])]

    return [{'text': component, 'entities': entities} for component, entities in zip(sub_components, updated_entities)]

    import copy


def clean_whitespaces(article):

    text = article['text'].strip()
    entities = copy.deepcopy(article['entities'])

    new_text = ''

    for idx, char in enumerate(text):
        if char == ' ' and text[idx+1] == ' ':
            # Update entities positions
            for entity in entities:
                if len(new_text) < entity['start_pos']:
                    entity['start_pos'] -= 1
                    entity['end_pos'] -= 1
        else:
            new_text += char

        # Stop if at last index position
        if idx == len(text) - 1:
            break

    return {'text': new_text, 'entities': entities}


def align_annotations(article):

    text = article['text']
    entities = copy.deepcopy(article['entities'])

    for top in entities:

        if text[top['start_pos']:top['end_pos']] != top['text']:

            start = text[:top['end_pos']+5].rfind(top['text'])

            end = start + len(top['text'])

            top['start_pos'] = start
            top['end_pos'] = end

    return {'text': text, 'entities': entities}
