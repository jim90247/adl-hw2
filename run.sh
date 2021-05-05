#!/bin/bash
#
# Run homework 2 tasks end-to-end.
set -e

function show_message() {
    echo "$(date +%Y-%m-%dT%H:%M:%S)|" "$@"
}

function abort_exec() {
    reason=$*
    show_message "ERROR:" "${reason}"
    exit 1
}

if [ "$#" -lt 3 ]; then
    abort_exec "${0} requires three arguments: dataset, context and prediction output."
fi

CONTEXT="$1"
DATASET="$2"
PREDICTION="$3"

RUN_ID="$(uuidgen)"

NS_TASKS="${NS_TASKS:-"predict"}"
NS_MODEL_CKPT="ckpt/ns-chinese-roberta-wwm-ext"
NS_INPUT_DATASET="${DATASET}_ns_preprocessed_${RUN_ID}.json"
NS_OUTPUT="$(pwd)/next_sentence_output_${RUN_ID}"

QA_TASKS="${QA_TASKS:-"predict"}"
QA_MODEL_CKPT="ckpt/qa-chinese-roberta-wwm-ext_full-train"
QA_INPUT_DATASET="${NS_OUTPUT}/predictions.json"
QA_OUTPUT="$(pwd)/question_answering_output_${RUN_ID}"

show_message "Tasks - Next sentence: ${NS_TASKS}"
show_message "Tasks - Question answering: ${QA_TASKS}"

function cleanup() {
    show_message "Cleaning up intermediate output..."
    rm -rf "${NS_OUTPUT}" "${QA_OUTPUT}" "${NS_INPUT_DATASET}"
}

# clean up intermediate output when exit
trap cleanup EXIT

show_message "Converting input format..."
python3.8 convert_next_sentence.py "${DATASET}" "${CONTEXT}" "${NS_INPUT_DATASET}"
show_message "Next sentence prediction input dataset is written to ${NS_INPUT_DATASET}."

show_message "Running next sentence prediction..."
python3.8 run_ns_wrapper.py "${NS_MODEL_CKPT}" \
    --context "${CONTEXT}" \
    --test_input "${NS_INPUT_DATASET}" \
    --raw_test_input "${DATASET}" \
    --output "${NS_OUTPUT}" \
    --tasks "${NS_TASKS}" \
    --overwrite
if [ ! -f "${QA_INPUT_DATASET}" ]; then
    abort_exec "run_ns_wrapper.py does not produce ${QA_INPUT_DATASET}."
fi
show_message "Question answering input is written to ${QA_INPUT_DATASET}"

show_message "Running question answering..."
python3.8 run_qa_wrapper.py "${QA_MODEL_CKPT}" \
    --ns_prediction "${QA_INPUT_DATASET}" \
    --output "${QA_OUTPUT}" \
    --tasks "${QA_TASKS}" \
    --overwrite
if [ ! -f "${QA_OUTPUT}/test_predictions.json" ]; then
    abort_exec "run_qa_wrapper.py does not produce ${QA_OUTPUT}/test_predictions.json."
fi
cp "${QA_OUTPUT}/test_predictions.json" "${PREDICTION}"
show_message "Prediction is written to ${PREDICTION}"
