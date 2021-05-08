import json
from random import sample


def split_train_val(input_dataset_path, train_output, val_output, frac=0.2):
    with open(input_dataset_path, encoding='utf-8') as f:
        input_dataset = json.load(f)

    val_dataset = sample(input_dataset, int(len(input_dataset) * frac))
    train_dataset = [data for data in input_dataset if data not in val_dataset]

    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=2, ensure_ascii=False)

    with open(val_output, 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, indent=2, ensure_ascii=False)


split_train_val('dataset/train.json', 'dataset/train_.json', 'dataset/eval.json')
