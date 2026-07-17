#!/usr/bin/env python3
from __future__ import annotations
import json, platform, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def command_output(args):
    try: return subprocess.run(args,capture_output=True,text=True,timeout=15).stdout.strip()
    except Exception: return "unavailable"

def collect():
    tools={name:shutil.which(name) or "unavailable" for name in ["nvidia-smi","nvcc","mpirun","mpiexec","nsys","all_reduce_perf","all_gather_perf","reduce_scatter_perf"]}
    torch_info={"installed":False}
    try:
        import torch
        torch_info={"installed":True,"version":torch.__version__,"cuda_available":torch.cuda.is_available(),"cuda_devices":torch.cuda.device_count(),"cuda_version":torch.version.cuda,"distributed":torch.distributed.is_available(),"nccl_backend":torch.distributed.is_nccl_available() if torch.distributed.is_available() else False}
    except Exception as exc: torch_info["error"]=str(exc)
    return {"timestamp_utc":datetime.now(timezone.utc).isoformat(),"platform":platform.platform(),"python":sys.version,"tools":tools,"torch":torch_info,"gpu_query":command_output([tools["nvidia-smi"],"-L"]) if tools["nvidia-smi"]!="unavailable" else "unavailable","git_commit":command_output(["git","rev-parse","HEAD"])}

if __name__=="__main__":
    output=Path(sys.argv[1] if len(sys.argv)>1 else "nccl_environment.json"); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(collect(),indent=2),encoding="utf-8"); print(output)
