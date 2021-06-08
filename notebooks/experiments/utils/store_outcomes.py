import pickle


def store_outcome(model_name, dataset, strict, forgiving):

    model_name = model_name.replace('/', '-')

    with open(f'outcomes-{model_name}-{dataset}.pkl', 'wb') as file:

        pickle.dump((strict, forgiving), file)


def read_outcome_pickle_file(model_name, dataset):

    model_name = model_name.replace('/', '-')

    with open(f'outcomes-{model_name}-{dataset}.pkl', 'rb') as file:
        outcome = pickle.load(file)

    print('Outcome format: \n')
    print('Tuple: (strict, forgiving)) --> fps, fns, tps')

    return outcome
