import copy


def calc_precision(tp, fp):
    return tp/(tp + fp)


def calc_recall(tp, fn):
    return tp/(tp + fn)


def calc_fscore(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def evaluate(gold_truth_labels, predictions):
    # Counts of true positives, false positives & false negatives
    tp, fp, fn = 0, 0, 0

    # List with false positives and false negatives
    fps, fns = [], []

    for gold, pred in zip(gold_truth_labels, predictions):

        tp_tmp, fp_tmp, fn_tmp, fns_temp, fps_temp = evaluate_one_article(
            gold, pred)

        tp += tp_tmp
        fp += fp_tmp
        fn += fn_tmp

        fns.extend(fns_temp)
        fps.extend(fps_temp)

    precision = calc_precision(tp, fp)
    recall = calc_recall(tp, fn)
    f_score = calc_fscore(precision, recall)

    print(f'fp: {fp} | tp: {tp} | fn: {fn}')
    print(
        f'precision: {precision:.3f} | recall: {recall:.3f} | f-score: {f_score:.3f}')

    return fps, fns


def evaluate_one_article(gold_truth, prediction):

    gold = gold_truth['entities'].copy()
    pred = prediction['entities'].copy()

    # Counts of true positives, false positives & false negatives
    tp, fp, fn = 0, 0, 0

    # List with false positives and false negatives
    fps, fns = [], []

    i = 0

    while len(gold) > 0 and len(pred) > 0:
        i += 1

        # Check if the first two elements are the same
        if gold[0] == pred[0]:
            tp += 1
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

    if len(gold) > 0:
        fn += 1
    elif len(pred) > 0:
        fp += 1

    return tp, fp, fn, fns, fps
