import json
from argparse import ArgumentParser
from typing import Dict, List

import tqdm
import random


def hw2_to_squad(dataset: List[Dict], contexts: List[str], include_answers: bool = True) -> List[Dict]:
    """Convert homework 2 dataset format to SQuAD 1.0 format.

    Args:
        dataset (List[Dict]): homework 2 dataset
        contexts (List[str]): homework 2 context file
        include_answers (bool, optional): Whether to include answers in output. Calculating metrics requires answers. Defaults to True.

    Raises:
        ValueError: some required fields do not exist in some dataset entry.

    Returns:
        List[Dict]: Homework 2 dataset in SQuAD 1.0 format.
    """
    squad_dataset = []
    # squad fields:
    # id: str
    # title: str
    # context: str
    # question: str
    # answer: {"answer_start": List[int], "text": List[str]}

    required_fields = ['id', 'question', 'relevant']

    for entry in tqdm.tqdm(dataset, desc="Converting dataset to SQuAD 1.0 format"):
        for field_name in required_fields:
            if entry.get(field_name) is None:
                raise ValueError("No {} in input dataset!".format(field_name))

        squad_entry = {
            'id': entry['id'],
            'question': entry['question'],
            'context': contexts[entry['relevant']],
            'answers': {
                'answer_start': [],
                'text': []
            }
        }

        if include_answers and entry.get('answers') is not None:
            for answer in entry['answers']:
                squad_entry['answers']['answer_start'].append(answer['start'])
                squad_entry['answers']['text'].append(answer['text'])

        squad_dataset.append(squad_entry)

    return squad_dataset


def main(args):
    with open(args.dataset, encoding='utf-8') as f:
        dataset = json.load(f)
    with open(args.contexts, encoding='utf-8') as f:
        contexts = json.load(f)

    squad_dataset = hw2_to_squad(dataset, contexts, include_answers=args.answer)

    if args.count is not None:
        squad_dataset = random.sample(squad_dataset, k=args.count)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({"data": squad_dataset}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('contexts')
    parser.add_argument('output')
    parser.add_argument('--count', type=int)
    parser.add_argument('--answer', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
