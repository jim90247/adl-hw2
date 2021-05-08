#!/bin/bash
set -xe

python3.8 -m spacy download zh_core_web_md

NS_MODEL_CKPT_TAR_URL="https://www.dropbox.com/s/ijxlrhsbgy0ke92/ns-chinese-roberta-wwm-ext.tar.gz?dl=1"
QA_MODEL_CKPT_TAR_URL="https://www.dropbox.com/s/nuflcdp28zrfqaf/qa-chinese-roberta-wwm-ext.tar.gz?dl=1"

NS_MODEL_CKPT_TAR="ns-chinese-roberta-wwm-ext.tar.gz"
QA_MODEL_CKPT_TAR="qa-chinese-roberta-wwm-ext.tar.gz"

[ -d ckpt ] || mkdir ckpt
cd ckpt

wget "${NS_MODEL_CKPT_TAR_URL}" -O "${NS_MODEL_CKPT_TAR}"
wget "${QA_MODEL_CKPT_TAR_URL}" -O "${QA_MODEL_CKPT_TAR}"

tar zxvf "${NS_MODEL_CKPT_TAR}"
tar zxvf "${QA_MODEL_CKPT_TAR}"
