# TODO

- [ ] `run.sh`
- [x] `download.sh`
- [ ] public and private predictions (optional)
- [ ] `README.md`
- [x] `report.pdf`

## Download script

Usage:

```bash
bash ./download.sh
```

File size should be less than 4G and should finish within 1 hour.

Subtasks

- [x] Upload model to Dropbox
- [x] Complete scripts

## Run script

Usage:

```bash
bash ./run.sh /path/to/context.json /path/to/public.json /path/to/pred/public.json
bash ./run.sh /path/to/context.json /path/to/private.json /path/to/pred/private.json
```

Note that `python` would be python 3.8.
Note that the execution environment has only 1 GPU with 8G RAM, and 20G disk space.

Subtasks

- [ ] Verify the training and testing process do not need network access
- [x] Complete scripts
- [x] Run the script in a clean environment to make sure it works

## README

Write down the command to train the model.

## Report

Subtasks

- [x] data processing
  - [x] describe how tokenizer works
  - [x] answer span
    - [x] How did you convert the answer span start/end position on characters to position on tokens after BERT tokenization?
    - [x] After your model predicts the probability of answer span start/end position, what rules did you apply to determine the final start/end position?
- [x] Modeling
  - [x] describe model and its performance, loss function used, optimization algorithm, learning rate and batch size
  - [x] another pretrained model (chinese-roberta-wwm-ext in my case) and its performance, the difference between the first model and this one
- [x] Curves (create a validation set from training dataset)
  - [x] em
  - [x] f1
- [x] Pretrained vs non-pretrained
  - [x] configuration of the model and how to train it
  - [x] performance of this model v.s. BERT
