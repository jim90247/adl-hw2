from collections import defaultdict
import json
from argparse import ArgumentParser
from typing import Dict, List

import tqdm
import random


def hw2_to_squad(dataset: List[Dict], contexts: List[str], no_answer_frac: float) -> List[Dict]:
    """Convert homework 2 dataset format to SQuAD 1.0 format.

    Args:
        dataset (List[Dict]): homework 2 dataset
        contexts (List[str]): homework 2 context file

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
            if field_name not in entry:
                raise ValueError("No {} in input dataset!".format(field_name))

        no_answer_count = int(no_answer_frac / (1 - no_answer_frac))
        paragraphs = random.sample([p for p in entry['paragraphs'] if p != entry['relevant']],
                                   k=min(no_answer_count, len(entry['paragraphs'])-1)) + [entry['relevant']]
        for context_id in paragraphs:
            squad_entry = {
                'id': entry['id'],
                'question': entry['question'],
                'context': contexts[context_id],
                # Answers will be left empty for non-relevant paragraphs.
                'answers': {
                    'answer_start': [],
                    'text': []
                }
            }

            if 'answers' in entry and context_id == entry['relevant']:
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

    if args.count is not None:
        dataset = random.sample(dataset, k=args.count)

    squad_dataset = hw2_to_squad(dataset, contexts, args.no_answer_frac)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({"data": squad_dataset}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('contexts')
    parser.add_argument('output')
    parser.add_argument('--count', type=int)
    parser.add_argument('--no_answer_frac', type=float, default=0.0)
    args = parser.parse_args()
    main(args)
