import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse
import pandas as pd
from torch import nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.ao.pruning import WeightNormSparsifier

SparseSemiStructuredTensor._FORCE_CUTLASS = False
torch.manual_seed(100)

CSV_COLUMNS = ["model", "lora_rank", "allocated", "max_allocated"]
# Load model
def conv1d_to_linear(conv1d):
    input_dim = conv1d.weight.shape[0]
    output_dim = conv1d.weight.shape[1]
    bias = conv1d.bias is not None
    layer = torch.nn.Linear(input_dim, output_dim, bias=bias)
    layer.weight.data = conv1d.weight.data.T
    if bias:
        layer.bias.data = conv1d.bias.data
    return layer


def prune_tensor(mat, N=2, M=4):
    reshaped_mat = mat.clone().reshape(-1, M)
    mask = torch.zeros_like(reshaped_mat, dtype=torch.bool)
    if (N, M) == (1, 2):
        _, indices = torch.topk(torch.abs(reshaped_mat), k=N, dim=1, sorted=False, largest=True)
        rows = (indices == 1).sum(dim=1)
        mask[:, 0] = rows
        mask[:, 1] = torch.logical_not(rows)
    elif (N, M) == (2, 4):
        _, indices = torch.topk(torch.abs(reshaped_mat), k=N, dim=1, sorted=False, largest=True)
        rows = torch.logical_not((indices == 0).sum(dim=1))
        mask[:, 0] = rows
        rows = torch.logical_not((indices == 1).sum(dim=1))
        mask[:, 1] = rows
        rows = torch.logical_not((indices == 2).sum(dim=1))
        mask[:, 2] = rows
        rows = torch.logical_not((indices == 3).sum(dim=1))
        mask[:, 3] = rows
    elif N < M / 2:
        _, indices = torch.topk(torch.abs(reshaped_mat), k=N, dim=1, sorted=False, largest=True)
        for i in range(M):
            rows = torch.logical_not((indices == i).sum(dim=1))
            mask[:, i] = rows
    else:
        _, indices = torch.topk(torch.abs(reshaped_mat), k=(M - N), dim=1, sorted=False, largest=False)
        for i in range(M):
            rows = (indices == i).sum(dim=1)
            mask[:, i] = rows
    reshaped_mat[mask] = 0.
    return reshaped_mat.reshape(mat.shape), mask.reshape(mat.shape)

def get_layers_list(model):
    if hasattr(model, "model"):
        layers = model.model.decoder.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        raise NotImplementedError
    return layers

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.compile
def to_sparse_semi_structured_compiled(x):
    return to_sparse_semi_structured(x)

def sparsify_model(model, rank, data_type):
    sparsifier = WeightNormSparsifier(
        # apply sparsity to all blocks
        sparsity_level=1.0,
        # shape of 4 elemens is a block
        sparse_block_shape=(1, 4),
        # two zeros for every block of 4
        zeros_per_block=2
    )

    layers = get_layers_list(model)
    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
        for lin in subset:
            mask = (
                torch.Tensor([0, 0, 1, 1])
                .tile((subset[lin].weight.shape[0], subset[lin].weight.shape[1] // 4))
                .half()
                .to("cuda")
                .bool()
            )
            #print(mask.dtype)
            if data_type =="int8":
                #subset[lin].weight = torch.nn.Parameter((mask * subset[lin].weight.to(torch.int8)), requires_grad=False)
                subset[lin].weight = torch.nn.Parameter((prune_tensor(subset[lin].weight)[0]).to(torch.int8), requires_grad=False)
            else:
                subset[lin].weight = torch.nn.Parameter((mask * subset[lin].weight))
            print(subset[lin].weight.data.dtype)

    mem = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"Mem: {mem:.3f}MB")

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
        for lin in subset:
            #print(type(subset[lin].weight))
            subset[lin].weight = nn.Parameter(to_sparse_semi_structured_compiled(subset[lin].weight), requires_grad=False)
            #print(subset[lin].weight.data.dtype)
            if data_type =="int8":
                subset[lin].lora_left = nn.Parameter(torch.randint(-10,120,(subset[lin].weight.data.shape[1],int(subset[lin].weight.data.shape[0]*rank)), dtype =torch.int8, device = subset[lin].weight.device ), requires_grad=False) 
                subset[lin].lora_left = nn.Parameter(torch.randint(-10,120,(int(subset[lin].weight.data.shape[0]*rank),subset[lin].weight.data.shape[0]), dtype = torch.int8, device = subset[lin].weight.device ), requires_grad=False)
            #print(subset[lin].lora_left.data.dtype)
            else:
                subset[lin].lora_left = nn.Parameter(torch.randn(subset[lin].weight.data.shape[1],int(subset[lin].weight.data.shape[0]*rank), dtype =torch.float16, device = subset[lin].weight.device)) 
                subset[lin].lora_left = nn.Parameter(torch.randn(int(subset[lin].weight.data.shape[0]*rank),subset[lin].weight.data.shape[0], dtype = torch.float16, device = subset[lin].weight.device ))
    mem = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"Mem: {mem:.3f}MB")

def to_int8(model):
    layers = get_layers_list(model)
    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
        for lin in subset:
            #subset[lin].weight.requires_grad_(False)
            #subset[lin].weight.data = subset[lin].weight.data.to(torch.int8)
            subset[lin].weight = torch.nn.Parameter((subset[lin].weight.to(torch.int8)), requires_grad=False)
            print(subset[lin].weight.data.dtype)

def get_llm(model_name, rank, cache_dir="llm_weights", local_checkpoint_dir="", data_type="fp16"):
    allocated_mem, max_mem =query_memory()

#    if '30b' in model_name or '66b' in model_name:
#        model = AutoModelForCausalLM.from_pretrained(
#        model_name,
#        torch_dtype=torch.float16,
#       cache_dir=cache_dir,
#    )
#    else:
    model_size = model_name.split('/')[1]
    last_dir = os.listdir(cache_dir+"/models--facebook--"+model_size+"/snapshots")[0]
    model = AutoModelForCausalLM.from_pretrained(
            cache_dir+"/models--facebook--"+model_size+"/snapshots/"+last_dir,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map='auto',
            local_files_only=True
        )
    
    #to_int8(model)
    if rank > 0:
        sparsify_model(model, rank, data_type)
    else:
     if data_type=="int8":
      to_int8(model)
    if local_checkpoint_dir != "":
        checkpoint = torch.load(local_checkpoint_dir, map_location="cpu")
        model.load_state_dict(checkpoint)

    for i, layer in enumerate(get_layers_list(model)):
        if hasattr(layer, "attn"):
            layer.attn.c_attn = conv1d_to_linear(layer.attn.c_attn)
            layer.attn.c_proj = conv1d_to_linear(layer.attn.c_proj)
        if hasattr(layer, "mlp"):
            layer.mlp.c_fc = conv1d_to_linear(layer.mlp.c_fc)
            layer.mlp.c_proj = conv1d_to_linear(layer.mlp.c_proj)

    print(model.device)
    allocated_mem, max_mem = query_memory()
    model.seqlen = model.config.max_position_embeddings
    return model, round(allocated_mem, 2),round(max_mem,2)

# Query the memory usage
def query_memory():
    #os.system('nvidia-smi')
    device = torch.device("cuda")
    memory_allocated = torch.cuda.memory_allocated(device)
    max_memory_allocated = torch.cuda.max_memory_allocated(device)

    print(f"Memory allocated: {memory_allocated / 1024 / 1024} MB")
    print(f"Max memory allocated: {max_memory_allocated / 1024 / 1024} MB")

    return memory_allocated / 1024 / 1024, max_memory_allocated / 1024 / 1024


def add_result_to_csv(args,  allocated_mem, max_mem):
    # Load CSV if it exists, otherwise create a new DataFrame with given columns
    if os.path.exists("out/cusfp.csv"):
        df = pd.read_csv("out/cusfp.csv")
    else:
        df = pd.DataFrame(columns=CSV_COLUMNS)

    # Check if the row combination exists and update perplexity
    new_row_data = {column: getattr(args, column) for column in CSV_COLUMNS[:-2]}
    row_exists = df.index[(df[CSV_COLUMNS[:-2]] == pd.Series(new_row_data)).all(axis=1)]

    # Now we don't mind adding perplexity
    new_row_data['allocated'] = allocated_mem
    new_row_data['max_allocated'] = max_mem

    if row_exists.empty:
        # Row combination does not exist, add a new row
        new_row_df = pd.DataFrame([new_row_data], columns=CSV_COLUMNS)
        df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        # Row combination exists, modify perplexity
        index_to_update = row_exists.values[0]
        df.at[index_to_update, 'allocated'] = new_row_data['allocated']
        df.at[index_to_update, 'max_allocated'] = new_row_data['max_allocated']

    # Save to CSV
    df.to_csv("out/cusfp.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='opt model')
    parser.add_argument("--lora_rank", type=float, default=0.0)
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument("--local_checkpoint_dir", type=str, default="")
    parser.add_argument("--data_type", type=str, default="fp16")    
    args = parser.parse_args()

    model, allocated_mem, max_mem = get_llm(args.model,args.lora_rank, args.cache_dir, args.local_checkpoint_dir, args.data_type)
    print(allocated_mem, max_mem)
    add_result_to_csv(args, allocated_mem, max_mem)

if __name__ == '__main__':
    main()
