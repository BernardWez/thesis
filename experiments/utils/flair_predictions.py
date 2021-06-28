from tqdm import tqdm
from flair.data import Sentence


def make_predictions(toponyms_data, tagger):

    predictions = []

    for article in tqdm(toponyms_data):

        text = article['text']

        # make a sentence
        sentence = Sentence(text)

        # run NER over sentence
        tagger.predict(sentence)

        pred = sentence.to_dict(tag_type='ner')
        pred['entities'] = [entity for entity in pred['entities']
                            if entity['labels'][0].value == 'LOC']
        [entity.pop('labels') for entity in pred['entities']]
        pred.pop('labels')

        predictions.append(pred)

    return predictions
