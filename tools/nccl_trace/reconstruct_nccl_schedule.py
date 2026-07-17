#!/usr/bin/env python3
from __future__ import annotations
import argparse
from drac_eval.collective_trace import write_collective_events_csv
from drac_eval.nccl_log import parse_nccl_log
from drac_eval.nccl_reconstruct import reconstruct_ring_schedule

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("log");p.add_argument("--output",required=True);a=p.parse_args();execution=reconstruct_ring_schedule(parse_nccl_log(a.log));write_collective_events_csv(a.output,[e for op in execution.operations for e in op.events])
