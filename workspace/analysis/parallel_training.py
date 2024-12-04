import os
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.functional import cross_entropy
from torcheval.metrics.functional import (
multiclass_accuracy, multiclass_auroc, multiclass_f1_score, multiclass_confusion_matrix)
from tqdm import tqdm
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def stacking_gpu(rank, world_size, list_tensor_to_stack, sampler_length):
    if len(list_tensor_to_stack[0].shape) == 2:
        total_tensor = torch.vstack(list_tensor_to_stack).to(rank)
        gather_tensor = torch.zeros((sampler_length * world_size, total_tensor.shape[1]), dtype=total_tensor.dtype, device=rank)
    else: 
        total_tensor = torch.hstack(list_tensor_to_stack).to(rank)
        gather_tensor = torch.zeros(sampler_length * world_size, dtype=total_tensor.dtype, device=rank)

    dist.all_gather_into_tensor(gather_tensor, total_tensor)
    del total_tensor, list_tensor_to_stack
    torch.cuda.empty_cache()
    return gather_tensor.cpu().detach()



def send_res(output, target, child_conn, mode="train", give_matrix=False):
    with torch.no_grad():
        child_conn.send((mode, "losses", cross_entropy(output, target).cpu().detach().numpy()))
        output = nn.Softmax(dim=1)(output)
        metrics = {"acc": multiclass_accuracy, 
                   "roc": multiclass_auroc, 
                   "f1": multiclass_f1_score}
        for (token, func) in list(metrics.items()):
            child_conn.send((mode, token, 
                             func(output, target, num_classes=output.shape[1], average=None).cpu().detach().numpy()))
        if give_matrix==True:
            child_conn.send((mode, "matrix", 
                             multiclass_confusion_matrix(output, target, num_classes=output.shape[1]).cpu().detach().numpy()))


def eval(rank, model, dataloader):
    model.eval()
    output_list, target_list = [], []
    with torch.no_grad():
        for data, target in dataloader:
            data_gpu = data.to(rank)
            output = model(data_gpu)
            output_list.append(output.cpu().detach())
            target_list.append(target.cpu().detach())
            del data, data_gpu, target, output
            torch.cuda.empty_cache()
    return output_list, target_list


def train(rank, world_size, model, model_param, adam_param, train_dataset, test_dataset,
          batch_size, epochs, fold,
          filename, allow_eval, child_conn):
    setup(rank, world_size)
    # Define model, loss function, and optimizer
    model = model(*model_param).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()#reduction='sum'
    optimizer = optim.Adam(ddp_model.parameters(), lr=adam_param["lr"], 
                           weight_decay=adam_param["weight_decay"])

    # Use DistributedSampler to handle distributed data loading
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    
    for epoch in (tqdm(range(epochs)) if rank==0 else range(epochs)):
        ddp_model.train()
        train_sampler.set_epoch(epoch)  # Ensure each epoch sees different data
        test_sampler.set_epoch(epoch)

        train_output_list, train_target_list = [], []
        for train_data, train_target in train_dataloader:
            train_data, train_target = train_data.to(rank), train_target.to(rank)
            optimizer.zero_grad()
            train_output = ddp_model(train_data)
            loss = criterion(train_output, train_target)
            loss.backward()
            optimizer.step()

            train_output_list.append(train_output.cpu().detach())
            train_target_list.append(train_target.cpu().detach())
            
            del train_data, train_target, train_output
            torch.cuda.empty_cache()

        if allow_eval==True:
            # train_output_list, train_target_list = eval(rank, ddp_model, train_dataloader)
            test_output_list, test_target_list = eval(rank, ddp_model, test_dataloader)

            train_gather_output = stacking_gpu(rank, world_size, train_output_list, len(train_sampler))
            train_gather_target = stacking_gpu(rank, world_size, train_target_list, len(train_sampler))
            test_gather_output = stacking_gpu(rank, world_size, test_output_list, len(test_sampler))
            test_gather_target = stacking_gpu(rank, world_size, test_target_list, len(test_sampler))
            
            if rank == 0:
                give_matrix = (False if epoch != (epochs - 1) else True)
                send_res(train_gather_output, train_gather_target, child_conn, mode="train", give_matrix=give_matrix)
                send_res(test_gather_output, test_gather_target, child_conn, mode="test", give_matrix=give_matrix)
    
            del train_gather_output, train_gather_target, test_gather_output, test_gather_target
            torch.cuda.empty_cache()
    
    if rank == 0:
        torch.save(ddp_model.module.state_dict(), "trained_model/"+filename+f"_fold_{fold}.pth")
    cleanup()




def run_train(model, model_param, adam_param, train_dataset, test_dataset,
              batch_size, epochs, fold, filename, allow_eval):
    world_size = torch.cuda.device_count()

    parent_conn, child_conn = mp.Pipe()
    dict_res = {"train": {"losses": [],
                          "acc": [],
                          "roc": [],
                          "f1": [],
                          "matrix": []},
                "test": {"losses": [],
                          "acc": [],
                          "roc": [],
                          "f1": [],
                          "matrix": []}}
    
    mp.spawn(train,
             args=(world_size, model, model_param, adam_param, train_dataset, test_dataset,
                   batch_size, epochs, fold, filename, allow_eval, child_conn),
             nprocs=world_size,
             join=True)

    while parent_conn.poll():
        (split, token, res) = parent_conn.recv()
        dict_res[split][token].append(res)

    for key in list(dict_res.keys()):
        dict_res[key]["losses"] = np.array(dict_res[key]["losses"])
        dict_res[key]["matrix"] = np.array(dict_res[key]["matrix"][0])
        for metric in ["acc", "roc", "f1"]:
           dict_res[key][metric] = np.vstack(dict_res[key][metric])
    dict_res = pd.DataFrame(dict_res)
    dict_res.to_pickle("trained_model/" + filename + f"_fold_{fold}_result.pkl")
        
    return dict_res

