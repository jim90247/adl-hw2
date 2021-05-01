import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List


def hw2_to_next_sentence(dataset: List[Dict], contexts: List[str]):
    for entry in dataset:
        for context_id in entry['paragraphs']:
            d = {
                'id': "{}~{}".format(entry['id'], context_id),
                'question': entry['question'],
                'context': contexts[context_id]
            }
            if 'relevant' in entry:
                d['label'] = int(entry['relevant'] != context_id)
            yield d


def main(args):
    with open(args.dataset, encoding='utf-8') as f:
        dataset = json.load(f)
        if args.count is not None:
            dataset = dataset[:args.count]
    with open(args.context, encoding='utf-8') as f:
        contexts = json.load(f)

    converted = list(hw2_to_next_sentence(dataset, contexts))
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({'data': converted}, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("context", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    main(args)
