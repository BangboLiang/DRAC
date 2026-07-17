#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from dataclasses import asdict
from drac_eval.nccl_log import parse_nccl_log

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("log");p.add_argument("--output",required=True);a=p.parse_args();record=parse_nccl_log(a.log)
    payload=asdict(record)
    with open(a.output,"w",encoding="utf-8") as handle:json.dump(payload,handle,indent=2)
