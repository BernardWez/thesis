import numpy as np
from evaluate import evaluate
from loading_functions import prepare_data
from preparing_dataset import prepare_dataset
from process_predictions import process_predictions
from transformers import Trainer
from transformers import DataCollatorForTokenClassification


def cross_validation_mBERT(model, tokenizer, label_list, fold):

    # DutchPolicyDocs
    file_path = '../../../data/DutchPolicyDocs/DutchPolicyDocs.json'
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
    evaluate(data_DPD, processed_results, model_name='mBERT',
             dataset=dataset, filtered=filtered, cv=fold)

    print('Yay')
