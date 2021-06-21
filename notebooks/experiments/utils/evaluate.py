import pandas as pd


def calc_precision(tp, fp):
    return tp/(tp + fp)


def calc_recall(tp, fn):
    return tp/(tp + fn)


def calc_fscore(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def calc_accuracy(tp, total_annotations):
    return tp / total_annotations


def evaluate(gold_truth_labels, predictions, model_name, dataset, filtered, cv=False):

    fps_strict, fns_strict, tps_strict = run_evaluation(
        gold_truth_labels, predictions, 'strict', model_name, dataset, filtered, cv)

    fps_forgiving, fns_forgiving, tps_forgiving = run_evaluation(
        gold_truth_labels, predictions, 'forgiving', model_name, dataset, filtered, cv)

    return ((fps_strict, fns_strict, tps_strict), (fps_forgiving, fns_forgiving, tps_forgiving))


def store_results(model_name, dataset, mode, precision,
                  recall, f_score, accuracy, filtered, cv):

    # Load results dataframe
    if cv:
        results = pd.read_pickle('../../../results/cv-results.pkl')

        # Add results to dataframe
        results = results.append({'model': model_name,
                                  'dataset': dataset,
                                  'mode': mode,
                                  'filtered': filtered,
                                  'fold': cv,
                                  'precision': precision,
                                  'recall': recall,
                                  'f1': f_score,
                                  'acc': accuracy}, ignore_index=True)

        # Remove duplicates
        results = results.drop_duplicates(
            ['model', 'dataset', 'mode', 'filtered', 'fold'], keep='last')

        # Write back to disk
        results.to_pickle("../../../results/cv-results.pkl")

    else:
        results = pd.read_pickle('../../../results/results.pkl')

        # Add results to dataframe
        results = results.append({'model': model_name,
                                  'dataset': dataset,
                                  'mode': mode,
                                  'filtered': filtered,
                                  'precision': precision,
                                  'recall': recall,
                                  'f1': f_score,
                                  'acc': accuracy}, ignore_index=True)

        # Remove duplicates
        results = results.drop_duplicates(
            ['model', 'dataset', 'mode', 'filtered'], keep='last')

        # Write back to disk
        results.to_pickle("../../../results/results.pkl")


def run_evaluation(gold_truth_labels, predictions, mode,
                   model_name, dataset, filtered, cv):
    # Counts of true positives, false positives & false negatives
    tp, fp, fn = 0, 0, 0

    # List with false positives, false negatives, and true positives
    fps, fns, tps = [], [], []

    for gold, pred in zip(gold_truth_labels, predictions):

        if mode == 'strict':
            tp_tmp, fp_tmp, fn_tmp, fns_temp, fps_temp, tps_temp = evaluate_one_article_strict(
                gold, pred)
        elif mode == 'forgiving':
            tp_tmp, fp_tmp, fn_tmp, fns_temp, fps_temp, tps_temp = evaluate_one_article_forgiving(
                gold, pred)

        tp += tp_tmp
        fp += fp_tmp
        fn += fn_tmp

        fns.extend(fns_temp)
        fps.extend(fps_temp)
        tps.extend(tps_temp)

    precision = calc_precision(tp, fp)
    recall = calc_recall(tp, fn)
    f_score = calc_fscore(precision, recall)

    total_annotations = sum([len(entry['entities'])
                            for entry in gold_truth_labels])
    accuracy = calc_accuracy(tp, total_annotations)

    if cv:
        # Auto overwrite results for cross validation
        overwrite = 'y'
    else:
        overwrite = input('Do you want to overwrite results? (y/n)')

    if overwrite == 'y':
        store_results(model_name, dataset, mode, precision,
                      recall, f_score, accuracy, filtered, cv)
    else:
        print('Results not saved...')

    print(f'Evaluation mode: {mode}')
    print(f'fp: {fp} | tp: {tp} | fn: {fn}')
    print(
        f'precision: {precision:.3f} | recall: {recall:.3f} | f-score: {f_score:.3f} | accuracy: {accuracy:.3f}')
    print('------------------------------------------------------------------------')
    print()

    return fps, fns, tps


def evaluate_one_article_strict(gold_truth, prediction):

    gold = gold_truth['entities'].copy()
    pred = prediction['entities'].copy()

    # Counts of true positives, false positives & false negatives
    tp, fp, fn = 0, 0, 0

    # List with false positives and false negatives
    fps, fns, tps = [], [], []

    while len(gold) > 0 and len(pred) > 0:
        # Check if the first two elements are the same
        if gold[0] == pred[0]:
            tp += 1
            tps.append(pred[0]['text'])

            gold.pop(0)
            pred.pop(0)

        else:
            # Grab the first appearing element
            element, source = (
                gold[0], 'gold') if gold[0]['start_pos'] < pred[0]['start_pos'] else (pred[0], 'pred')

            # Remove the element first appearing element
            if source == 'gold':
                fn += 1
                fns.append(element['text'])
                gold.remove(element)
            elif source == 'pred':
                fp += 1
                fps.append(element['text'])
                pred.remove(element)

    if gold:
        for i in range(len(gold)):
            fn += 1
            fns.append(gold[0]['text'])
            gold.pop(0)

    elif pred:
        for i in range(len(pred)):
            fp += 1
            fps.append(pred[0]['text'])
            pred.pop(0)

    return tp, fp, fn, fns, fps, tps


def evaluate_one_article_forgiving(gold_truth, prediction):

    gold = gold_truth['entities'].copy()
    pred = prediction['entities'].copy()

    # Counts of true positives, false positives & false negatives
    tp, fp, fn = 0, 0, 0

    # List with false positives and false negatives
    fps, fns, tps = [], [], []

    while gold and pred:

        # Check if the first two elements have overlap
        if (
            # case 1: prediction starts before gold annotation and ends after gold start
            (pred[0]['start_pos'] < gold[0]['start_pos']
             and pred[0]['end_pos'] > gold[0]['start_pos'])

            or

            # case 2: prediction is within gold annotation boundaries
            (pred[0]['start_pos'] >= gold[0]['start_pos']
             and pred[0]['end_pos'] <= gold[0]['end_pos'])

            or

            # case 3: prediction starts before gold end and ends at/after gold end
            (pred[0]['start_pos'] < gold[0]['end_pos']
             and pred[0]['end_pos'] >= gold[0]['end_pos'])

        ):

            tp += 1
            tps.append(pred[0]['text'])
            pred.pop(0)

            # Break if last predictions has been handeled
            if not pred:
                gold.pop(0)
                break
            # elif 'next prediction' starts before gold annotation ends: skip to next iteration
            elif pred[0]['start_pos'] < gold[0]['end_pos']:
                continue
            # Else 'next prediction' starts after: remove gold annotation
            else:
                gold.pop(0)

        else:
            # Grab the first appearing element
            element, source = (
                gold[0], 'gold') if gold[0]['start_pos'] < pred[0]['start_pos'] else (pred[0], 'pred')

            # Remove the element first appearing element
            if source == 'gold':
                fn += 1
                fns.append(element['text'])
                gold.remove(element)
            elif source == 'pred':
                fp += 1
                fps.append(element['text'])
                pred.remove(element)

    if gold:
        for i in range(len(gold)):
            fn += 1
            fns.append(gold[0]['text'])
            gold.pop(0)

    elif pred:
        for i in range(len(pred)):
            fp += 1
            fps.append(pred[0]['text'])
            pred.pop(0)

    return tp, fp, fn, fns, fps, tps
