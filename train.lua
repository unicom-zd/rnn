NUM_OF_TR_AT = 2110176
NUM_OF_TR_LT =  93827
NUM_OF_FEAT = 23
NUM_OF_DAY = 31
NUM_OF_CLASS = 2 -- AT = 1, LT = 2
CLASS_AT = 1.0
CLASS_LT = 2.0
BAT_SIZE = 10
AT_LT_NORM = {0, 0.1, 0.5} -- mean,std,shift

math = require 'math'

tr_at = torch.Tensor(NUM_OF_TR_AT, NUM_OF_DAY, NUM_OF_FEAT):fill(1)
tr_lt = torch.Tensor(NUM_OF_TR_LT, NUM_OF_DAY, NUM_OF_FEAT):fill(2)

function get_at_lt_split(norm, bs)
    local mean, std, shift = table.unpack(norm)
    local al_lt_frac = torch.normal(mean, std) + shift
    local al_lt_split = torch.round(bs * al_lt_frac)
    al_lt_split = math.max(1, al_lt_split)
    al_lt_split = math.min(bs-1, al_lt_split)
    return al_lt_split
end

function load_at(data, shuf_at ,input, target, at_id, at_lt_split, bs)
    for sub_at_id = at_id, at_id + at_lt_split - 1 do
        local bat_id = sub_at_id - at_id + 1
        local shuf_at_id = shuf_at[sub_at_id]
        input[bat_id] = data[shuf_at_id]
        target[bat_id] = CLASS_AT
    end
    at_id = at_id + al_lt_split
    return at_id
end

function load_lt(data, shuf_lt ,input, target, lt_id, at_lt_split, bs)
    for sub_lt_id = lt_id, lt_id + bs - at_lt_split - 1 do
        local bat_id = at_lt_split + sub_lt_id - lt_id + 1
        local shuf_lt_id = shuf_lt[sub_lt_id]
        input[bat_id] = data[shuf_lt_id]
        target[bat_id] = CLASS_LT
    end
    lt_id = lt_id + bs - at_lt_split
    return lt_id
end

function train()
    local shuf_at = torch.randperm(NUM_OF_TR_AT)
    local shuf_lt = torch.randperm(NUM_OF_TR_LT)
    -- local num_of_used_tr_at = NUM_OF_TR_AT-(NUM_OF_TR_AT%BAT_SIZE)
    -- local shuf_tr_at = shuf_tr_at[{{1,num_of_used_tr_at}}]
    
    local lt_id = 1
    local at_id = 1
    
    while at_id < NUM_OF_TR_AT - BAT_SIZE do
        local input = torch.Tensor(BAT_SIZE, NUM_OF_DAY, NUM_OF_FEAT)
        local target = torch.Tensor(BAT_SIZE)
        
        local at_lt_split = get_at_lt_split(AT_LT_NORM, BAT_SIZE)
        
        -- load at
        at_id = load_at(tr_at, shuf_at ,input, target, at_id, at_lt_split, BAT_SIZE)
        
        -- reshuffle lt and start from 0 if not enough left for a batch
        if lt_id + BAT_SIZE - at_lt_split - 1 > NUM_OF_TR_LT then
            lt_id = 1
            shuf_lt = torch.randperm(NUM_OF_TR_LT)
        end
        
        -- load lt
        lt_id = load_lt(tr_lt, shuf_lt ,input, target, lt_id, at_lt_split, BAT_SIZE)
        
        -- assert((at_lt_split*CLASS_AT+(BAT_SIZE-at_lt_split)*CLASS_LT)/BAT_SIZE == torch.mean(input), 'wrong')
        -- assert(torch.mean(input) == torch.mean(target), 'wrong')
    end
    -- print('pass')
end

-- train()
