from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .collective_trace import CollectiveEvent, aggregate_ordered_demand


def direction_metrics(matrix: np.ndarray) -> Dict[str,float]:
    a=v=0.0; active=0
    for i in range(matrix.shape[0]):
        for j in range(i+1,matrix.shape[1]):
            x,y=float(matrix[i,j]),float(matrix[j,i]); a+=abs(x-y); v+=x+y
            active += int(x>0)+int(y>0)
    return {"absolute_directionality_bytes":a,"traffic_bytes":v,"omega":a/v if v>0 else float("nan"),"active_direction_count":active}


def build_mapping(rank_count:int, level:str, placement:str, hierarchy:Dict[str,int], seed:int, actual_hosts:Dict[int,str]|None=None)->np.ndarray:
    if placement=="actual" and actual_hosts:
        labels={host:i for i,host in enumerate(sorted(set(actual_hosts.values())))}
        return np.array([labels[actual_hosts[r]] for r in range(rank_count)],dtype=int)
    size={"rank":1,"server":hierarchy["gpus_per_server"],"tor":hierarchy["gpus_per_server"]*hierarchy["servers_per_tor"],"aggregation":hierarchy["gpus_per_server"]*hierarchy["servers_per_tor"]*hierarchy["tors_per_aggregation"]}[level]
    groups=max(1,int(np.ceil(rank_count/size)))
    if placement=="contiguous": return np.minimum(np.arange(rank_count)//size,groups-1)
    if placement=="round_robin": return np.arange(rank_count)%groups
    if placement=="random":
        labels=np.minimum(np.arange(rank_count)//size,groups-1); np.random.default_rng(seed).shuffle(labels); return labels
    raise ValueError(placement)


def aggregate_matrix(matrix:np.ndarray,mapping:np.ndarray)->np.ndarray:
    groups=sorted(set(mapping.tolist())); index={v:i for i,v in enumerate(groups)}; out=np.zeros((len(groups),len(groups)))
    for i in range(len(mapping)):
        for j in range(len(mapping)):
            if mapping[i]!=mapping[j]: out[index[int(mapping[i])],index[int(mapping[j])]]+=matrix[i,j]
    return out


def weighted_direction_jaccard(left:np.ndarray,right:np.ndarray)->Tuple[float,int]:
    intersection=union=0.0; flips=0
    for i in range(left.shape[0]):
        for j in range(i+1,left.shape[1]):
            ld=1 if left[i,j]>=left[j,i] else -1; rd=1 if right[i,j]>=right[j,i] else -1
            lw=abs(float(left[i,j]-left[j,i])); rw=abs(float(right[i,j]-right[j,i]))
            if lw>0 and rw>0 and ld!=rd: flips+=1
            if ld==rd: intersection+=min(lw,rw)
            union+=max(lw,rw)
    return (intersection/union if union>0 else float("nan"),flips)


def analyze_timescales(events:Sequence[CollectiveEvent],rank_count:int,windows:Sequence[int],hierarchy:Dict[str,int],placements:Sequence[str],seed:int)->Tuple[List[Dict[str,object]],List[Dict[str,object]],List[Dict[str,object]],List[Dict[str,object]]]:
    step_keys=[]
    for event in events:
        key=(event.phase,event.step)
        if key not in step_keys: step_keys.append(key)
    step_mats=[aggregate_ordered_demand([e for e in events if (e.phase,e.step)==key],rank_count) for key in step_keys]
    times=[]; levels=[]; persistence=[]; flips=[]
    rank_total=sum(step_mats,np.zeros((rank_count,rank_count)))
    rank_v=direction_metrics(rank_total)["traffic_bytes"]
    for window in windows:
        mats=[sum(step_mats[start:start+window],np.zeros_like(step_mats[0])) for start in range(0,len(step_mats),window)]
        for idx,matrix in enumerate(mats):
            times.append({"window_steps":window,"window_index":idx,**direction_metrics(matrix),"provenance":"NCCL-SCHEDULE-DERIVED"})
        for idx in range(1,len(mats)):
            value,count=weighted_direction_jaccard(mats[idx-1],mats[idx]); persistence.append({"window_steps":window,"left_window":idx-1,"right_window":idx,"direction_persistence":value,"provenance":"NCCL-SCHEDULE-DERIVED"}); flips.append({"window_steps":window,"window_transition":idx-1,"dominant_direction_flip_count":count,"provenance":"NCCL-SCHEDULE-DERIVED"})
    for placement in placements:
        for level in ["rank","server","tor","aggregation"]:
            mapping=build_mapping(rank_count,level,placement,hierarchy,seed)
            aggregate=aggregate_matrix(rank_total,mapping); metrics=direction_metrics(aggregate)
            levels.append({"placement":placement,"level":level,**metrics,"boundary_traffic_fraction":metrics["traffic_bytes"]/rank_v if rank_v>0 else float("nan"),"provenance":"NCCL-SCHEDULE-DERIVED"})
    return times,levels,persistence,flips
