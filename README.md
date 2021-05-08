# Applied Deep Learning Homework 2

## Training

First convert provided dataset `train.json` into `ns-train.json` and `qa-train.json` with following commands:

```bash
# convert to next sentence input format
python3.8 convert_next_sentence.py /path/to/context.json /path/to/train.json dataset/ns-train.json
# convert to question answering input format
python3.8 convert_squad.py /path/to/context.json /path/to/train.json dataset/qa-train.json
```

Then use `run.sh` to perform training and prediction tasks.
Setting `NS_TASKS` and `QA_TASKS` environment variables to configure which tasks to run.

```bash
# second parameter `/path/to/public.json` can be changed to `/path/to/private.json` to generate private test dataset prediction.
NS_TASKS=train,predict QA_TASKS=train,predict run.sh /path/to/context.json /path/to/public.json /path/to/prediction.json
```
