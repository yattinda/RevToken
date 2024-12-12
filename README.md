# RevToken

This repository is the replication package including the source code and a part of datasets.

NOTE：This repository excluded files that is too large to upload to Github
## Source code
```
|
|-Openstack
|
|-QtBase
|
|-replication-package(previous study)
```

## require
Show
```
requirements.txt
```

## RUN
Operation confirmed on only Linux

(if you need) run ```/train/train.py``` to create model.
### RQ1. line-level prediction
Run ```eval_for_linelevel/test.py```

### RQ2. token-level prediction
Run ```eval_for_tokenlevel/attention/attention_test.py``` or ```eval_for_tokenlevel/lime/lime_test.py```

if you want to run *N or more token*, you change param *more_than_n* in line 20 to 30

To build your own model, please download the entire replication package from zenodo [here](https://zenodo.org/records/14160121)
