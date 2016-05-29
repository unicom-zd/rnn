local function load_data(name)
    local hdf5 = require 'hdf5'
    local options = hdf5.DataSetOptions()
    options:setDeflate()
    return hdf5.open(name..'.h5', 'r'):read(name, options):all()
end

local function get_atlt_split(norm, bs)
    local mean, std, shift = table.unpack(norm)
    local atlt_frac = torch.normal(mean, std) + shift
    local atlt_split = torch.round(bs * atlt_frac)
    atlt_split = math.max(1, atlt_split)
    atlt_split = math.min(bs-1, atlt_split)
    return atlt_split
end

local function resample(at_dataloader, lt_dataloader, split, bs)
    local longTensor = torch.LongTensor
    local at_indices = longTensor(split):random(1, at_dataloader:size())
    local lt_indices = longTensor(bs-split):random(1, lt_dataloader:size())
    local at_inputs, at_targets = at_dataloader:index(at_indices)
    local lt_inputs, lt_targets = lt_dataloader:index(lt_indices)
    local input = torch.cat({at_inputs, lt_inputs}, 1)
    local target = torch.cat({at_targets, lt_targets}, 1)
    return input, target
end

local function clone (t) -- deep-copy a table
    if type(t) ~= "table" then return t end
    local meta = getmetatable(t)
    local target = {}
    for k, v in pairs(t) do
        if type(v) == "table" then
            target[k] = clone(v)
        else
            target[k] = v
        end
    end
    setmetatable(target, meta)
    return target
end
local function clean_userdata(t)
    local copy_t = clone(t)
    for k, v in pairs(copy_t) do
        if type(v) == 'table' then
            copy_t[k] = clean_userdata(v)
        elseif type(v) == 'userdata' then
            copy_t[k] = nil
        end
    end
    return copy_t
end
local function dump_para(name)
    local cjson = require 'cjson'
    local file = io.open(name..'.json', 'w')
    local data = {}
    for k,v in pairs(_G) do
        if k:upper() == k and k:sub(1,1) ~= '_' then
            data[k] = v
        end
    end
    local clean_data = clean_userdata(data)
    local txt = cjson.encode(clean_data)
    file:write(txt)
    file:close()
end
local function load_para(filename)
    local file = io.open(filename..'.json', 'r')
    local line = file:read()
    file:close()
    local cjson = require 'cjson'
    local data = cjson.decode(line)
    for k,v in pairs(data) do
        _G[k] = v
    end
end
local function clear_para()
    for k,v in pairs(_G) do
        if k:upper() == k and k:sub(1,1) ~= '_' then
            _G[k] = nil
        end
    end
end
local function print_para()
    for k,v in pairs(_G) do
        if k:upper() == k and k:sub(1,1) ~= '_' then
            print(k, v)
        end
    end
end

return {
    load_data = load_data,
    get_atlt_split = get_atlt_split,
    resample = resample,
    load_para = load_para,
    dump_para = dump_para,
    clear_para = clear_para,
    print_para = print_para,
}
