import copy
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Union

import datasets
import numpy as np
import transformers
from datasets.arrow_dataset import Dataset
from transformers import (
    AutoConfig, AutoModelForNextSentencePrediction, AutoTokenizer, DataCollatorWithPadding, EvalPrediction,
    HfArgumentParser, PreTrainedTokenizerFast, Trainer, TrainingArguments, default_data_collator, set_seed
)
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_utils import (PredictionOutput, get_last_checkpoint, is_main_process)
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


@dataclass
class HomeworkDatasetArguments:
    """
    Arguments containing paths to original homework 2 datasets.
    """

    raw_test_file: str = field(default='./dataset/private.json', metadata={'help': 'Homework 2 original test dataset'})

    context_file: str = field(default='./dataset/context.json', metadata={'help': 'Homework 2 context file'})

    qa_file: str = field(default='./squad-input.json', metadata={'help': 'Input file name for question answering task'})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help":
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help":
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help":
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
                "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "For debugging purposes or quicker training, truncate the number of test examples to this "
                "value if set."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None and self.train_file is None and self.validation_file is None and
            self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


def run_next_sentence(args_dict: Union[Dict, None] = None):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, HomeworkDatasetArguments))
    if args_dict is not None:
        # Allow main function be called with arguments from other python codes
        model_args, data_args, training_args, hw2_data_args = parser.parse_dict(args_dict)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, hw2_data_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, hw2_data_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load homework 2 dataset from json files
    datasets = load_dataset(data_args)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForNextSentencePrediction.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    preprocessed_datasets = preprocess_dataset(datasets, tokenizer, data_args, training_args)

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator if data_args.pad_to_max_length else
        DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    def compute_metrics(eval_pred: EvalPrediction):
        def raw_accuracy(eval_pred: EvalPrediction) -> float:
            pred_labels = np.argmax(eval_pred.predictions, axis=1)
            return sum(pred_labels == eval_pred.label_ids) / len(pred_labels)

        return {'raw_accuracy': raw_accuracy(eval_pred)}

    train_dataset = preprocessed_datasets['train'] if 'train' in preprocessed_datasets else None
    eval_dataset = preprocessed_datasets['validation'] if 'validation' in preprocessed_datasets else None
    test_dataset = preprocessed_datasets['test'] if 'test' in preprocessed_datasets else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Unless we are not training, there's no need to run evaluation again here.
    # Evaluation will be run after each training epoch.
    if training_args.do_eval and not training_args.do_train:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        raw_predictions = trainer.predict(
            copy.deepcopy(test_dataset)
        )  # trainer.predict() removes columns that model.forward() do not accept
        final_predictions = get_final_predictions(raw_predictions, test_dataset)

        with open(hw2_data_args.raw_test_file, encoding='utf-8') as f:
            dataset = json.load(f)

        with open(hw2_data_args.context_file, encoding='utf-8') as f:
            contexts = json.load(f)

        qa_input = list(dump_predictions(final_predictions, dataset, contexts))
        with open(hw2_data_args.qa_file, 'w', encoding='utf-8') as f:
            json.dump({'data': qa_input}, f, ensure_ascii=False, indent=2)


def load_dataset(data_args: DataTrainingArguments) -> Dict[str, Dataset]:
    # Training and validation datasets have labels, but test dataset does not.
    # Therefore, we load them seperately, otherwise `load_dataset` will report errors.
    data_files_with_labels = {}
    if data_args.train_file is not None:
        data_files_with_labels["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files_with_labels["validation"] = data_args.validation_file

    datasets_with_answers = datasets.load_dataset('json', data_files=data_files_with_labels,
                                                  field="data") if len(data_files_with_labels) > 0 else {}

    data_files_without_labels = {}
    if data_args.test_file is not None:
        data_files_without_labels["test"] = data_args.test_file
    datasets_without_answers = datasets.load_dataset('json', data_files=data_files_without_labels,
                                                     field="data") if len(data_files_without_labels) > 0 else {}

    return {**datasets_with_answers, **datasets_without_answers}


def preprocess_dataset(
    dataset_dict: Dict[str, Dataset], tokenizer: PreTrainedTokenizerFast, data_args: DataTrainingArguments,
    training_args: TrainingArguments
) -> Dict[str, Dataset]:
    """Preprocess datasets.

    Args:
        dataset_dict (Dict[str, Dataset]): Datasets loaded from files.
        tokenizer (PreTrainedTokenizerFast): The tokenizer.
        data_args (DataTrainingArguments): The data training arguments.
        training_args (TrainingArguments): The training arguments.

    Returns:
        Dict[str, Dataset]: Preprocessed datasets.
    """

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def prepare_features(examples) -> BatchEncoding:
        """Tokenize and preprocess features."""
        tokenized_examples = tokenizer(
            examples['question'],
            examples['context'],
            truncation='only_second' if pad_on_right else 'only_first',
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            padding="max_length" if data_args.pad_to_max_length else False
        )
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        for column in ['label', 'id']:
            if column in examples:
                tokenized_examples[column] = [examples[column][i] for i in sample_mapping]
        return tokenized_examples

    preprocessed_dataset_dict = {}
    for do_split, split, max_samples in zip(
        [training_args.do_train, training_args.do_eval, training_args.do_predict], ['train', 'validation', 'test'],
        [data_args.max_train_samples, data_args.max_val_samples, data_args.max_test_samples]
    ):
        if not do_split:
            continue
        if split not in dataset_dict:
            raise ValueError(f"Missing {split} dataset")
        examples = dataset_dict[split]
        column_names = examples.column_names

        if max_samples is not None:
            examples = examples.select(range(max_samples))

        examples = examples.map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        preprocessed_dataset_dict[split] = examples if max_samples is None else examples.select(max_samples)

    return preprocessed_dataset_dict


def get_final_predictions(raw_predictions: PredictionOutput, preprocessed_dataset: Dataset) -> Dict[str, str]:
    scores = defaultdict(lambda: ('', float('-inf')))  # {qid: (context_id, score)}
    for raw_pred, example in zip(raw_predictions.predictions, preprocessed_dataset):
        qid, context_id = example['id'].split('~')
        score = raw_pred[0] - raw_pred[1]  # is_next - not_next
        if score > scores[qid][1]:
            scores[qid] = (context_id, score)
    return {qid: p[0] for qid, p in scores.items()}


def dump_predictions(predictions: Dict[str, str], dataset: List[Dict], contexts: List[str]):
    for entry in dataset:
        qid = entry['id']
        ctx_id = int(predictions[qid])
        yield {'id': qid, 'question': entry['question'], 'context': contexts[ctx_id]}
