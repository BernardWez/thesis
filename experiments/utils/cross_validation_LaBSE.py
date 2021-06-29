import numpy as np
from evaluate import evaluate
from loading_functions import prepare_data
from preparing_dataset import prepare_dataset
from process_predictions import process_predictions
from transformers import Trainer
from transformers import DataCollatorForTokenClassification

# NOTE: Executed in Google Colab


def cross_validation_LaBSE(model, tokenizer, label_list, fold):
    """
    Evaluates trained LaBSE model on all four evaluation corpora and writes the results to file stored in Google
    """

    # DutchPolicyDocs
    file_path = 'drive/MyDrive/Thesis/data/DutchPolicyDocs.json'
    filtered = False
    dataset = 'DutchPolicyDocs'

    # Load data
    data_DPD = prepare_data(file_path, filtered=filtered, split=True)

    # Process data
    DPD = prepare_dataset(data_DPD, tokenizer)

    # Evaluation trainer
    data_collator = DataCollatorForTokenClassification(tokenizer)

    eval_trainer = Trainer(model,
                           data_collator=data_collator)

    # Predictions
    raw_pred, _, _ = eval_trainer.predict(DPD)
    predictions = np.argmax(raw_pred, axis=2)

    processed_results = process_predictions(
        predictions, DPD, label_list, tokenizer)

    # Evaluate
    evaluate(data_DPD, processed_results, model_name='LaBSE',
             dataset=dataset, filtered=filtered, cv=fold)

    # TR-News
    file_path = 'drive/MyDrive/Thesis/data/TR-News.xml'
    filtered = False
    dataset = 'TR-News'

    # Load data
    data_TRN = prepare_data(file_path, filtered=filtered, split=True)

    # Process data
    TRN = prepare_dataset(data_TRN, tokenizer)

    # Evaluation trainer
    data_collator = DataCollatorForTokenClassification(tokenizer)

    eval_trainer = Trainer(model,
                           data_collator=data_collator)

    # Predictions
    raw_pred, _, _ = eval_trainer.predict(TRN)
    predictions = np.argmax(raw_pred, axis=2)

    processed_results = process_predictions(
        predictions, TRN, label_list, tokenizer)

    # Evaluate
    evaluate(data_TRN, processed_results, model_name='LaBSE',
             dataset=dataset, filtered=filtered, cv=fold)

    # LGL
    file_path = 'drive/MyDrive/Thesis/data/LGL.xml'
    filtered = False
    dataset = 'LGL'

    # Load data
    data_LGL = prepare_data(file_path, filtered=filtered, split=True)

    # Process data
    LGL = prepare_dataset(data_LGL, tokenizer)

    # Evaluation trainer
    data_collator = DataCollatorForTokenClassification(tokenizer)

    eval_trainer = Trainer(model,
                           data_collator=data_collator)

    # Predictions
    raw_pred, _, _ = eval_trainer.predict(LGL)
    predictions = np.argmax(raw_pred, axis=2)

    processed_results = process_predictions(
        predictions, LGL, label_list, tokenizer)

    # Evaluate
    evaluate(data_LGL, processed_results, model_name='LaBSE',
             dataset=dataset, filtered=filtered, cv=fold)

    # GeoWebNews
    file_path = 'drive/MyDrive/Thesis/data/GeoWebNews.xml'
    filtered = True
    dataset = 'GWN'

    # Load data
    data_GWN = prepare_data(file_path, filtered=filtered, split=True)

    # Process data
    GWN = prepare_dataset(data_GWN, tokenizer)

    # Evaluation trainer
    data_collator = DataCollatorForTokenClassification(tokenizer)

    eval_trainer = Trainer(model,
                           data_collator=data_collator)

    # Predictions
    raw_pred, _, _ = eval_trainer.predict(GWN)
    predictions = np.argmax(raw_pred, axis=2)

    processed_results = process_predictions(
        predictions, GWN, label_list, tokenizer)

    # Evaluate
    evaluate(data_GWN, processed_results, model_name='LaBSE',
             dataset=dataset, filtered=filtered, cv=fold)
