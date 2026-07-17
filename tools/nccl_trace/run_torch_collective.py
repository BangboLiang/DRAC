#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,socket,time
import torch
import torch.distributed as dist

def main():
    p=argparse.ArgumentParser();p.add_argument("--collective",choices=["allreduce","allgather","reducescatter","broadcast"],required=True);p.add_argument("--bytes",type=int,required=True);p.add_argument("--iterations",type=int,default=20);a=p.parse_args()
    if not torch.cuda.is_available() or not dist.is_nccl_available(): raise RuntimeError("CUDA-enabled PyTorch with NCCL backend is required")
    dist.init_process_group("nccl");rank=dist.get_rank();world=dist.get_world_size();device=torch.device("cuda",int(os.environ.get("LOCAL_RANK",0)));torch.cuda.set_device(device)
    count=max(1,a.bytes//4);tensor=torch.ones(count,device=device);times=[]
    for iteration in range(a.iterations+5):
        torch.cuda.synchronize();start=time.perf_counter()
        if a.collective=="allreduce":dist.all_reduce(tensor)
        elif a.collective=="allgather":dist.all_gather([torch.empty_like(tensor) for _ in range(world)],tensor)
        elif a.collective=="reducescatter":dist.reduce_scatter(torch.empty_like(tensor),[tensor.clone() for _ in range(world)])
        else:dist.broadcast(tensor,0)
        torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1e6
        if iteration>=5:times.append(elapsed)
    print(json.dumps({"rank":rank,"host":socket.gethostname(),"gpu":torch.cuda.get_device_name(device),"collective":a.collective,"bytes":a.bytes,"world_size":world,"runtime_us_mean":sum(times)/len(times),"provenance":"measured_runtime_log"}))
    dist.destroy_process_group()
if __name__=="__main__":main()
