from datasets import Dataset


def prepare_dataset(toponyms, tokenizer):
    """
    Prepares dataset as a HuggingFace dataset and performs tokenization.
    """

    list_input_data = [i['text'] for i in toponyms]

    dataset = Dataset.from_dict({'tokens': list_input_data})

    dataset = dataset.map(tokenizer, input_columns='tokens',
                          batched=True, fn_kwargs={'truncation': True})

    return dataset
