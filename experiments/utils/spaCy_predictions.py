from tqdm import tqdm


def make_predictions(ner_pipeline, data):

    predictions = []

    texts = [article['text'] for article in data]

    for doc in tqdm(ner_pipeline.pipe(texts)):
        # Do something with the doc here

        pred = {'entities': [{'text': ent.text,
                              'start_pos': 1 + len(doc[:ent.start].text),
                              'end_pos': len(ent.text) + 1 + len(doc[:ent.start].text)} for ent in doc.ents if
                             ent.label_ == 'LOC' or ent.label_ == 'GPE']}

        if pred:
            pred['text'] = doc.text

            predictions.append(pred)

    return predictions
