#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from pprint import pprint
from typing import Union


def json2utf8(input_path: Union[Path, str], output_path: Union[Path, str, None]):
    with open(input_path) as f:
        data = json.load(f)

    pprint(data)

    if output_path is not None:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    json2utf8(args.input, args.output)
