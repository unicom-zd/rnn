NUM_OF_TR_AT = 2110176
NUM_OF_TR_LT = 93827
NUM_OF_FEAT = 23
NUM_OF_DAY = 31
NUM_OF_CLASS = 2 -- AT = 1, LT = 2
CLASS_AT = 1.0
CLASS_LT = 2.0
BAT_SIZE = 10
ATLT_NORM = {0, 0.1, 0.5} -- mean,std,shift

math = require 'math'
dl = require 'dataload'

tr_at = torch.Tensor(NUM_OF_TR_AT, NUM_OF_DAY, NUM_OF_FEAT):fill(1)
tr_at_target = torch.Tensor(NUM_OF_TR_AT):fill(1)
tr_lt = torch.Tensor(NUM_OF_TR_LT, NUM_OF_DAY, NUM_OF_FEAT):fill(2)
tr_lt_target = torch.Tensor(NUM_OF_TR_LT):fill(2)

at_dataloader = dl.TensorLoader(tr_at, tr_at_target)
lt_dataloader = dl.TensorLoader(tr_lt, tr_lt_target)

function get_atlt_split(norm, bs)
    local mean, std, shift = table.unpack(norm)
    local atlt_frac = torch.normal(mean, std) + shift
    local atlt_split = torch.round(bs * atlt_frac)
    atlt_split = math.max(1, atlt_split)
    atlt_split = math.min(bs-1, atlt_split)
    return atlt_split
end

function resample(at_dataloader, lt_dataloader, bs)
    local split = get_atlt_split(ATLT_NORM, bs)
    local longTensor = torch.LongTensor
    local at_indices = longTensor(split):random(1, at_dataloader:size())
    local lt_indices = longTensor(bs-split):random(1, lt_dataloader:size())
    local at_inputs, at_targets = at_dataloader:index(at_indices)
    local lt_inputs, lt_targets = lt_dataloader:index(lt_indices)
    local input = torch.cat({at_inputs, lt_inputs}, 1)
    local target = torch.cat({at_targets, lt_targets}, 1)
    return input, target, split
end

function train(batchsize, epochsize)
    for i = 1, math.floor(epochsize / batchsize) do
        local input, target, atlt_split = resample(at_dataloader, lt_dataloader, batchsize)
        input = input:cuda()
        target = target:cuda()
    end
end
