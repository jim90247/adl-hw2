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
        'train_file': str(args.dataset_root / 'qa-train.json'),
        'validation_file': str(args.dataset_root / 'qa-eval.json'),
        'test_file': str(args.ns_prediction),
        'output_dir': args.output,
        'overwrite_output_dir': args.overwrite,
        'logging_dir': args.output / 'tensorboard_output',
        # logging, evaluation, saving will be conduct every gradient_accumulation_steps * xxx_step steps
        'gradient_accumulation_steps': 1,
        'logging_strategy': 'steps',
        'logging_steps': 300,
        # do_eval is True if evaluation_strategy is not 'no'
        'evaluation_strategy': 'steps' if 'eval' in args.tasks else 'no',
        'eval_steps': 300,
        'save_strategy': 'steps',
        'save_steps': 600,
        'save_total_limit': 3,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 128,
        'learning_rate': args.lr,
        'num_train_epochs': args.epoch,
        'max_seq_length': 384,
        'doc_stride': 128,
        'fp16': args.fp16,
        'version_2_with_negative': False
    }

    import run_qa
    run_qa.main(param_dict)

    # Save wrapper parameters
    with open(args.output / 'wrapper_param.json', 'w') as f:
        str_param_dict = {k: str(v) for k, v in param_dict.items()}
        json.dump(str_param_dict, f, indent=2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("model", help="Name of the model (download from huggingface) or path to local model directory.")
    parser.add_argument("--tasks", help="Comma seperated list of tasks to perform.", required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path('dataset'))
    parser.add_argument("--ns_prediction", type=Path, help="Path to next_sentence prediction output.", required=True)
    parser.add_argument("-o", "--output", help="output directory", required=True, type=Path)
    parser.add_argument(
        "--overwrite", help="Overwrite output directory contents when it exists.", action='store_true', default=False
    )
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("-n", "--epoch", type=int, default=2)
    parser.add_argument("--fp16", action='store_true', default=False)
    args = parser.parse_args()

    args.tasks = args.tasks.split(',')
    if not all(task in TASKS for task in args.tasks):
        raise ValueError(f"{args.tasks} contains unknown tasks.")

    main(args)
