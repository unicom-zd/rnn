# Data preprocessing

1. Sort all csv by the first column with 
  ```bash
  for f in *; do echo $f; sort -t"," -k1n,1 $f > sorted-$f; done;
  ```

2. Move all sorted file to a new directory and change the directory to this directory

3. Make sure all sorted csv from the same user group have the same first column
  ```bash
  # check column 1 is the same, take ZDJM_3GD_02 as an example
  for f in *_LT_* ; do echo $f; diff <(cat sorted-ZDJM_3GD_02_201510_LT_BILL.csv | cut -d ',' -f 1) <(cat $f | cut -d ',' -f 1); done;
  for f in *_AT_* ; do echo $f; diff <(cat sorted-ZDJM_3GD_02_201510_AT_BILL.csv | cut -d ',' -f 1) <(cat $f | cut -d ',' -f 1); done;
  ```

4. Check presented status with the following command and modify `used_st` in `script/preprocess.ipynb` according to the presented status
  ```bash
  # the output is the frequency of all presented status
  cat *|cut -d , -f 1 --complement|tr "," "\n"|sort|uniq -c
  ```

5. Run [preprocess.ipynb](http://nbviewer.jupyter.org/github/unicom-zd/rnn/blob/master/script/preprocess.ipynb)(`script/preprocess.ipynb`).

# Training RNN
Run training script:
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

## Training monitor
Monitor the training process and result with [monitor.ipynb](http://nbviewer.jupyter.org/github/unicom-zd/rnn/blob/master/script/monitor.ipynb)(`script/check.ipynb`).
