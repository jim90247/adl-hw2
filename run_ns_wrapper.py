import json
from argparse import ArgumentParser
from pathlib import Path

TASKS = ['train', 'eval', 'predict']


def main(args):
    param_dict = {
        'model_name_or_path': args.model,
        'do_train': 'train' in args.tasks,
        'do_eval': 'eval' in args.tasks,
        'do_predict': 'predict' in args.tasks,
        # Datasets in preprocessed format
        'train_file': str(args.dataset_root / 'ns-train.json'),
        'validation_file': str(args.dataset_root / 'ns-eval.json'),
        'test_file': str(args.test_input),
        # Datasets in original format
        'context_file': str(args.context),
        'raw_train_file': str(args.dataset_root / 'train_.json'),
        'raw_eval_file': str(args.dataset_root / 'eval.json'),  # validation dataset, contains labels
        'raw_test_file': str(args.raw_test_input),  # test dataset
        'output_dir': args.output,
        'qa_file': args.output / 'predictions.json',
        'overwrite_output_dir': args.overwrite,
        'logging_dir': args.output / 'tensorboard_output',
        # logging, evaluation, saving will be conduct every gradient_accumulation_steps * xxx_step steps
        'gradient_accumulation_steps': int(96 / (args.per_device_train_batch_size * args.ngpus)),
        'logging_strategy': 'steps',
        'logging_steps': 500,
        # do_eval is True if evaluation_strategy is not 'no'
        'evaluation_strategy': 'steps' if 'eval' in args.tasks else 'no',
        'eval_steps': 500,
        'save_strategy': 'no',
        'save_steps': 1000,
        'save_total_limit': 3,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'per_device_eval_batch_size': 128,
        'learning_rate': args.lr,
        'num_train_epochs': args.epoch,
        'max_seq_length': 512,
        'doc_stride': 128,
    }

    import run_ns
    run_ns.run_next_sentence(param_dict)

    # Save wrapper parameters
    with open(args.output / 'wrapper_param.json', 'w') as f:
        str_param_dict = {k: str(v) for k, v in param_dict.items()}
        json.dump(str_param_dict, f, indent=2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("model", help="Name of the model (download from huggingface) or path to local model directory.")
    parser.add_argument("--tasks", help="Comma seperated list of tasks to perform.", required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path('dataset'))
    parser.add_argument("--context", help="Path to context file", type=Path, required=True)
    parser.add_argument("--test_input", help="Path to test dataset", type=Path, required=True)
    parser.add_argument("--raw_test_input", help="Path to unpreprocessed test dataset", type=Path, required=True)
    parser.add_argument("-o", "--output", help="output directory", required=True, type=Path)
    parser.add_argument(
        "--overwrite", help="Overwrite output directory contents when it exists.", action='store_true', default=False
    )
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("-n", "--epoch", type=int, default=2)
    parser.add_argument("--ngpus", type=int, default=6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    args = parser.parse_args()

    args.tasks = args.tasks.split(',')
    if not all(task in TASKS for task in args.tasks):
        raise ValueError(f"{args.tasks} contains unknown tasks.")
    if args.ngpus < 1:
        raise ValueError("GPU number should at least 1.")
    if args.per_device_train_batch_size < 1:
        raise ValueError("Per GPU batch size should be at least 1.")

    main(args)
