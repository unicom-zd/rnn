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

## 预处理生成的数据格式

1. 预处理生成的矩阵为`离网或在网用户人数x数据天数xfeature维数`，现feature维数为24，其中前7维为7个连续变量，后17维为离散status转换而成的17维one hot vector，不出现的状态为-1，出现的状态为1。
2. 数据保存格式为`hdf5`。
3. 对连续变量做了中心化，即每个用户的各项连续数据先减去该用户所有天的该连续数据的均值，再除以该用户所有天的该连续数据的标准差。
4. 命名规则说明：如`1602_2m_AT_24pr.h5`说明为16年2月份在网用户的预处理后数据，包含两月即11跟12月，其中feature维数为24。

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
可用`nohup`来置于后台运行。

## 训练脚本的注意事项
1. 对于调试中的参数，可将其添加到命令行的参数，例如：`cmd:option('--MODEL_PREFIX', 'm1', 'model prefix')`，该函数将`MODEL_PREFIX`变量添加到命令行参数，其默认值为`'m1'`，第三个变量为参数说明
2. `RESAMPLE_RATIO`为数据resample的比例，如该变量为2，说明每个mini-batch中约为2/3为在网用户，1/3为离网用户。resample所采用公式如下，每个mini-batch在网用户数等于`max(min((Normal(mean＝0,std＝0.1)+RESAMPLE_RATIO/(RESAMPLE_RATIO+1))*batchsize,batchsize-1),1)`Normal为按正态分布取样。离网用户数则为batchsize减在网用户数。以上逻辑在[util.lua](https://github.com/unicom-zd/rnn/blob/master/util.lua#L8-L15)的`get_atlt_split(norm, bs)`中实现。
3. resample过程在[util.lua](https://github.com/unicom-zd/rnn/blob/master/util.lua#L17-L26)的`resample(at_dataloader, lt_dataloader, split, bs)`中实现，如需要使用3个月的日数据进行实验，只需改动该函数的实现及数据预处理脚步即可。


## Training monitor
Monitor the training process and result with [monitor.ipynb](http://nbviewer.jupyter.org/github/unicom-zd/rnn/blob/master/script/monitor.ipynb)(`script/check.ipynb`).

# MISC

## SSH

```bash
ssh yanglingling@202.116.86.116 -L127.0.0.1:1234:127.0.0.1:9999
```
