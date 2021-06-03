from tqdm import tqdm


def make_predictions(ner_pipeline, data):

    predictions = []

    texts = [article['text'] for article in data]

    for doc in tqdm(ner_pipeline.pipe(texts)):
        # Do something with the doc here

        pred = {'entities': [{'text': ent.text,
                              'start_pos': len(doc[0:ent.end].text) - len(doc[ent.start]),
                              'end_pos': len(doc[0:ent.end].text)} for ent in doc.ents if
                             ent.label_ == 'LOC' or ent.label_ == 'GPE']}

        if pred:
            pred['text'] = doc

            predictions.append(pred)

    return predictions
