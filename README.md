# data preprocessing
See [preprocess.ipynb](http://nbviewer.jupyter.org/github/unicom-zd/rnn/blob/master/script/preprocess.ipynb)(`script/preprocess.ipynb`).

# training RNN
```
Example:
th train.lua --MODEL_PREFIX "m2" --RESAMPLE_RATIO 4
Options:
  --MODEL_PREFIX   model prefix [m1]
  --RESAMPLE_RATIO AT:LT = RESAMPLE_RATIO:1 [1]
  --LT_WEIGHT      weight for predicted as LT, weight of AT is 1 [1]
  --NUM_OF_ITER    number of train iteration [10]
  --OPTIM_METHOD   optimization method, sgd or adam [sgd]
```

## training monitor
See [monitor.ipynb](http://nbviewer.jupyter.org/github/unicom-zd/rnn/blob/master/script/monitor.ipynb)(`script/check.ipynb`).
