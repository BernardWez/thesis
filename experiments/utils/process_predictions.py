def process_predictions(predictions, dataset, label_list, tokenizer):

    # Get initial results
    results = convert_predictions(
        predictions, dataset['input_ids'], label_list, tokenizer)

    # Intermediate results: remove invalid results
    results_int = [{'text': result['text'], 'entities': [
        entity for entity in result['entities'] if '#' not in entity['text']]} for result in results]

    # Finialize results
    processed_results = process_pred_results(results_int, dataset['tokens'])

    return processed_results


def convert_predictions(prediction_label_ids, tokenized_input_ids,
                        label_list, tokenizer):

    prediction_labels = [[label_list[p] for t, p in zip(
        tokens, pred) if t != 101 and t != 102] for tokens, pred in zip(tokenized_input_ids, prediction_label_ids)]

    tokens = [tokenizer.convert_ids_to_tokens(
        i, skip_special_tokens=True) for i in tokenized_input_ids]

    predictions = []

    for token_set, label_set in zip(tokens, prediction_labels):

        text = tokenizer.convert_tokens_to_string(token_set)

        pred = {'text': text, 'entities': []}

        adjust_start_pos = 0

        for idx in range(len(token_set)):
            if label_set[idx] == 'B-LOC' or label_set[idx] == 'I-LOC':

                if idx == len(label_set)-1:
                    pass

                # Case 1: B-LOC followed by I-LOC --> CONTINUE
                elif label_set[idx+1] == 'I-LOC':
                    adjust_start_pos += 1
                    continue

                # Case 2: B-LOC followed by other B-LOC (together) --> CONTINUE
                elif label_set[idx+1] == 'B-LOC' and '#' in token_set[idx+1]:
                    adjust_start_pos += 1
                    continue

                current_pos = idx
                toponym_tokens = tokenizer.convert_tokens_to_string(
                    token_set[current_pos-adjust_start_pos:current_pos+1])
                sub_sentence = tokenizer.convert_tokens_to_string(
                    token_set[:current_pos+1])
                end = len(sub_sentence)
                start = end - len(toponym_tokens)

                pred['entities'].append(
                    {'text': toponym_tokens, 'start_pos': start, 'end_pos': end})

                adjust_start_pos = 0

        predictions.append(pred)

    return predictions


def process_pred_results(pred_results, original_text_inputs):

    final_results = [align_pred_and_original_text(pred_result, original_text)
                     for pred_result, original_text in zip(pred_results, original_text_inputs)]

    return final_results


def align_pred_and_original_text(pred_result, original_text):

    pred_text = pred_result['text']

    idx = 0
    removed_indices = []
    add_indices = []

    while pred_text != original_text:

        char_post, char_original = pred_text[idx], original_text[idx]

        if char_post != char_original:

            if char_original == ' ':
                pred_text = pred_text[:idx] + ' ' + pred_text[idx:]

                add_indices.append(idx)

                continue

            pred_text = pred_text[:idx] + pred_text[idx+1:]

            removed_indices.append(idx)

            if idx > len(pred_text) - 1:
                break

            continue

        idx += 1

        if idx > len(pred_text) - 1:
            break

    pred_entities = pred_result['entities']

    for entity in pred_entities:
        for index in removed_indices:
            if index > entity['start_pos']:
                break
            else:
                entity['start_pos'] -= 1
                entity['end_pos'] -= 1

    for entity in pred_entities:
        for index in add_indices:
            if index > entity['start_pos']:
                break
            else:
                entity['start_pos'] += 1
                entity['end_pos'] += 1

    return {'text': pred_text, 'entities': pred_entities}
