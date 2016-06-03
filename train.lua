
MODEL_PREFIX = 'm6'
MODEL_SUFFIX = 's1'
PREFIX = MODEL_PREFIX .. '-' .. MODEL_SUFFIX
USE_BN = true
NUM_OF_FEAT = 23
NUM_OF_DAY = 31
NUM_OF_CLASS = 2 -- AT = 1, LT = 2
CLASS_AT = 1.0
CLASS_LT = 2.0
BAT_SIZE = 10
RESAMPLE_RATIO = 4
ATLT_NORM = {0, 0.1, RESAMPLE_RATIO/(RESAMPLE_RATIO+1)} -- mean,std,shift
-- {0, 0.1, 0.5} 1:1
-- {0, 0.1, 0.8} 4:1
HID_SIZE = 64
RHO = 999999 -- 30
BAT_SIZE = 256
DROPOUT = 0 -- 0.5 disable dropout when 0
LEARNING_RATE = 0.01
OPTIM_METHOD = 'adam'
OPTIM_PARA = {}
OPTIM_PARA['adam'] = {
    learningRate=LEARNING_RATE,
    beta1=0.9,
    beta2=0.995,
    epsilon=1e-7
}
OPTIM_PARA['sgd'] = {
    learningRate=LEARNING_RATE,
}

optim = require 'optim'
nn = require 'nn'
rnn = require 'rnn'
math = require 'math'
dataload = require 'dataload'
util = require 'util'
require 'nngraph'
require 'cunn'

torch.manualSeed(torch.initialSeed())

-- 1. load train data
tr_lt = util.load_data('tr_1m_LT_23pr')
tr_at = util.load_data('tr_1m_AT_23pr')
print(#tr_lt, #tr_at)
NUM_OF_TR_LT = (#tr_lt)[1]
NUM_OF_TR_AT = (#tr_at)[1]

tr_at_target = torch.Tensor(NUM_OF_TR_AT):fill(CLASS_AT)
tr_lt_target = torch.Tensor(NUM_OF_TR_LT):fill(CLASS_LT)
at_dataloader = dataload.TensorLoader(tr_at, tr_at_target)
lt_dataloader = dataload.TensorLoader(tr_lt, tr_lt_target)

-- 2. load val data
val_lt = util.load_data('val_1m_LT_23pr')
val_at = util.load_data('val_1m_AT_23pr')
print(#val_lt, #val_at)
NUM_OF_VAL_LT = (#val_lt)[1]
NUM_OF_VAL_AT = (#val_at)[1]

val_at_target = torch.Tensor(NUM_OF_VAL_AT):fill(CLASS_AT)
val_lt_target = torch.Tensor(NUM_OF_VAL_LT):fill(CLASS_LT)
val_at_dataloader = dataload.TensorLoader(val_at, val_at_target)
val_lt_dataloader = dataload.TensorLoader(val_lt, val_lt_target)

-- 3. define model
model = nn.Sequential()
do
    nn.FastLSTM.usenngraph = true -- faster
    nn.FastLSTM.bn = USE_BN

    local lstm = nn.Sequential()

    lstm:add(nn.FastLSTM(NUM_OF_FEAT, HID_SIZE, RHO))
    if DROPOUT > 0 then
        lstm:add(nn.Dropout(DROPOUT))
    end

    lstm:add(nn.FastLSTM(HID_SIZE, HID_SIZE, RHO))
    if DROPOUT > 0 then
        lstm:add(nn.Dropout(DROPOUT))
    end

    lstm = nn.Sequencer(lstm)

    model:add(nn.SplitTable(1, 2)) -- assuming batchSize x seqLen x feat
    model:add(lstm)
    model:add(nn.SelectTable(-1)) -- select last output
    model:add(nn.Linear(HID_SIZE, NUM_OF_CLASS))
end

-- 4. define criterion
criterion = nn.CrossEntropyCriterion()

-- 5. set up cuda
model = model:cuda()
criterion = criterion:cuda()

-- 6. auto-grad
parameters,gradParameters = model:getParameters()

function train(dl, batchsize, num_of_batch, prefix)
    print('begin trainning')
    model:training()
    util.dump_para(prefix..'-parameters')
    local time_logger = optim.Logger(prefix..'-time.log')
    time_logger:setNames{'setup time', 'for-back time', 'batch time'}
    local para_logger = optim.Logger(prefix..'-para.log')
    para_logger:setNames{'atlt split'}
    local train_logger = optim.Logger(prefix..'-train.log')
    train_logger:setNames{'training error'}

    local num_of_batch = num_of_batch or math.floor(NUM_OF_TR_AT / batchsize)
    local at_dataloader, lt_dataloader = table.unpack(dl)
    for i = 1, num_of_batch do
        local batch_timer = torch.Timer()

        -- prepare data
        local timer = torch.Timer()
        local split = util.get_atlt_split(ATLT_NORM, batchsize)
        local input, target = util.resample(at_dataloader, lt_dataloader, split, batchsize)
        input = input:cuda()
        target = target:cuda()
        local setup_time = timer:time().real

        -- forward + backward
        timer:reset()
        optim[OPTIM_METHOD](function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()

            model:forward(input)
            local loss = criterion:forward(model.output, target)

            criterion:backward(model.output, target)
            model:backward(input, criterion.gradInput)

            return loss, gradParameters
        end, parameters, OPTIM_PARA[OPTIM_METHOD])
        local bf_time = timer:time().real

        local batch_time = batch_timer:time().real
        train_logger:add{criterion.output}
        para_logger:add{split}
        time_logger:add{setup_time, bf_time, batch_time}
    end
    model:clearState()
    torch.save(prefix..'-model.t7',model)
end

function eval(dl, batchsize, prefix)
    print('begin eval')
    model:evaluate()
    local eval_logger = optim.Logger(prefix..'-eval.log')
    eval_logger:setNames{'at->at', 'at->lt', 'lt->at', 'lt->lt'}
    local confusion = optim.ConfusionMatrix(NUM_OF_CLASS)
    local at_dataloader, lt_dataloader = table.unpack(dl)
    local function loop(dataloader)
        local upbound = dataloader:size() - dataloader:size() % batchsize
        for k, inputs, targets in dataloader:subiter(batchsize, upbound) do
            local input = inputs:cuda()
            local target = targets:cuda()
            confusion:batchAdd(model:forward(input), target)
        end
    end
    loop(at_dataloader)
    loop(lt_dataloader)
    confusion:updateValids()
    local m = confusion.mat
    eval_logger:add{m[{1,1}], m[{1,2}], m[{2,1}], m[{2,2}]}
    return confusion
end

NUM_OF_BAT_IN_1EPOCH = math.floor((NUM_OF_TR_AT * 2) / BAT_SIZE)
START_ITER = 1
NUM_OF_ITER = 3
LEARNING_RATE = 0.01
OPTIM_PARA[OPTIM_METHOD]['learningRate'] = LEARNING_RATE
for i = START_ITER, START_ITER + NUM_OF_ITER - 1 do
    print('===it'..i..'===')
    local prefix = PREFIX .. '-it' .. tostring(i)
    train({at_dataloader,lt_dataloader}, BAT_SIZE,
            NUM_OF_BAT_IN_1EPOCH, prefix)
    local conf = eval({at_dataloader,lt_dataloader}, BAT_SIZE, prefix..'-train')
    print('train eval', conf.valids)
    local conf = eval({val_at_dataloader,val_lt_dataloader}, BAT_SIZE, prefix..'-vali')
    print('vali eval', conf.valids)
end

-- conf = eval({val_at_dataloader,val_lt_dataloader}, BAT_SIZE, PREFIX .. '-it1')
-- print(conf.valids)
