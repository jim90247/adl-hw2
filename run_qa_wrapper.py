from argparse import ArgumentParser
from pathlib import Path


def main(args):
    param_dict = {
        'model_name_or_path': args.model_path if args.model_name is None else args.model_name,
        'do_train': True,
        'do_eval': True,
        'do_predict': True,
        'train_file': str(args.dataset_root / 'qa-train.json'),
        'validation_file': str(args.dataset_root / 'qa-eval.json'),
        'test_file': str(args.ns_prediction),
        'output_dir': args.output,
        'overwrite_output_dir': args.overwrite,
        'logging_dir': args.output / 'tensorboard_output',
        # 'logging_strategy': 'step',
        # 'logging_steps': 500,
        # 'evaluation_strategy': 'step',  # do_eval is True if evaluation_strategy is not 'no'
        # 'eval_steps': 500,
        'save_total_limit': 3,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 384,
        'learning_rate': args.lr,
        'num_train_epochs': args.epoch,
        'max_seq_length': 384,
        'doc_stride': 128,
    }

    import run_qa
    run_qa.main(param_dict)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_name", help="Model name. Should not be used together with model_path.")
    parser.add_argument("--model_path", help="Model path. Should not be used together with model_name.")
    parser.add_argument("--dataset_root", type=Path, default=Path('dataset'))
    parser.add_argument("-o", "--output", help="output directory", required=True, type=Path)
    parser.add_argument(
        "--overwrite", help="Overwrite output directory contents when it exists.", action='store_true', default=False
    )
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("-n", "--epoch", type=int, default=2)
    parser.add_argument("--ns_prediction", type=Path, help="Path to next_sentence prediction output.", required=True)
    args = parser.parse_args()

    # Check if argument combination is valid
    if args.model_name is None and args.model_path is None:
        raise ValueError("Must specify one of model_name or model_path")
    elif args.model_name is not None and args.model_path is not None:
        raise ValueError("model_name and model_path cannot be specified at the same time")

    main(args)
