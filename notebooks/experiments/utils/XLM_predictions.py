from tqdm import tqdm


def make_predictions(ner_pipeline, articles):

    # Make the predictions
    predictions = [ner_pipeline(entry['text']) for entry in tqdm(articles)]

    # Filter out all non-location predictions
    location_predictions = [
        [pred for pred in entry if pred['entity'] == 'LOC'] for entry in predictions]

    # Format the entity predictions
    entity_predictions = []

    for entry_idx, entry in enumerate(location_predictions):

        # Entity list
        entities = []

        # Entity tracking variables
        text, start, end = '', 0, 0

        for idx, location in enumerate(entry):

            if not start:
                start = location['start']

            end = location['end']

            text = articles[entry_idx]['text'][start:end]

            # If last location, add to entities list
            if idx == len(entry) - 1:

                # Append entity to entities list
                entities.append(
                    {'text': text, 'start_pos': start, 'end_pos': end})

                # Reset entity tracking variables
                text, start, end = '', 0, 0

            # Check if together with next location, if not add to entities list
            elif location['end'] != entry[idx+1]['start']:

                # Append entity to entities list
                entities.append(
                    {'text': text, 'start_pos': start, 'end_pos': end})

                # Reset entity tracking variables
                text, start, end = '', 0, 0

            else:
                continue

        entity_predictions.append(entities)

    # Clean up the underscores
    for entry in entity_predictions:
        for toponym in entry:
            toponym['text'] = toponym['text'].replace('▁', ' ')
            if toponym['text'][0] == ' ':
                toponym['text'] = toponym['text'][1:]
                toponym['start_pos'] += 1

    # Merge into processed predictions
    processed_predictions = [{'text': text['text'], 'entities': entities} for text, entities in
                             zip(articles, entity_predictions)]

    return processed_predictions
