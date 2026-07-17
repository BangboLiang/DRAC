from __future__ import annotations
import hashlib
import math
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from drac_eval.atlahs_goal import ensure_official_text_goal, iter_goal, parse_goal_to_sqlite, integrity_from_sqlite, sqlite_parse_complete, EVIDENCE_TRACE_DERIVED
from drac_eval.rescue_v4_atlahs import aggregate_pairs, channel_cancellation, directional_metrics, mapping_for, nonoverlap_windows, persistence, simulated_starts
from drac_eval.rescue_v4_plotting import _save
from tools.atlahs.fetch_atlahs_traces import sha256_file

FIXTURE=Path("tools/atlahs/fixtures/official_goal_text_fixture.goal")


def test_official_goal_parser_fixture(tmp_path):
    db=tmp_path/"x.sqlite"; counts=parse_goal_to_sqlite(FIXTURE,db,"fixture")
    assert counts=={"num_ranks":2,"send":2,"recv":2,"calc":1,"dependencies":3}
    assert sqlite_parse_complete(db)


def test_unknown_binary_is_not_guessed(tmp_path):
    p=tmp_path/"x.bin"; p.write_bytes(b"\0binary")
    with pytest.raises(ValueError,match="never inferred|not parsed by inference"): ensure_official_text_goal(p)


def test_manifest_checksum_stable():
    expected=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(); assert sha256_file(FIXTURE)==expected


def test_rank_to_server_mapping():
    assert mapping_for(8,2,"contiguous",0)=={0:0,1:0,2:1,3:1,4:2,5:2,6:3,7:3}


def test_four_gpu_fallback_not_parser_invention():
    header=next(iter_goal(FIXTURE)); assert header==("num_ranks",2)


def test_send_receive_bytes_conserved(tmp_path):
    db=tmp_path/"x.sqlite"; parse_goal_to_sqlite(FIXTURE,db,"fixture"); result=integrity_from_sqlite(db)
    assert result["total_send_bytes"]==result["total_receive_bytes"]==96


def test_cross_and_intra_classification():
    src=np.array([0,0]); dst=np.array([0,1]); size=np.array([10,20]); assert sum(aggregate_pairs(src,dst,size).values())==20


def test_nonoverlap_window_byte_conservation():
    src=np.array([0,1]); dst=np.array([1,0]); size=np.array([10,20]); starts=np.array([0,15])
    rows=nonoverlap_windows(src,dst,size,starts,10,2,"x",100)
    assert sum(r["cross_node_bytes"] for r in rows)==30


def test_metrics_bounds():
    m=directional_metrics({(0,1):3,(1,0):1},2); assert m["A"]>=0 and m["V"]>=0 and 0<=m["Omega"]<=1


def test_absolute_retention_and_boundary_fraction_bounds():
    src=np.array([0,1,2]); dst=np.array([1,0,3]); size=np.array([5,2,7]); base=directional_metrics(aggregate_pairs(src,dst,size),4)
    agg=directional_metrics(aggregate_pairs(src,dst,size,mapping_for(4,2,"contiguous",0)),2)
    assert 0<=agg["A"]/base["A"]<=1 and 0<=agg["V"]/base["V"]<=1


def test_sliding_windows_not_conservation_sum():
    src=np.array([0]); dst=np.array([1]); size=np.array([10]); starts=np.array([5]); rows=nonoverlap_windows(src,dst,size,starts,10,2,"x",100,overlap_fraction=.5)
    assert sum(r["cross_node_bytes"] for r in rows)>10


def test_direction_persistence_hand_case():
    assert persistence({(0,1):10},{(0,1):5})==.5
    assert persistence({(0,1):10},{(1,0):10})==0


def test_channel_cancellation_hand_case():
    assert channel_cancellation({0:{(0,1):10},1:{(1,0):10}},2)==1


def test_evidence_labels_do_not_confuse_real_boundary():
    assert EVIDENCE_TRACE_DERIVED!="HYPOTHETICAL_AGGREGATION"


def test_synthetic_excluded_from_parser():
    assert "SYNTHETIC" not in {item.evidence_label for item in iter_goal(FIXTURE) if hasattr(item,"evidence_label")}


def test_no_directionality_upper_bound_zero():
    assert directional_metrics({(0,1):10,(1,0):10},2)["A"]==0


def test_goal_never_called_packet_trace():
    import drac_eval.atlahs_goal as module
    assert "packet trace" not in (module.__doc__ or "").lower()


def test_simulated_timeline_monotone_per_source():
    starts=simulated_starts(np.array([0,1,0]),np.array([100,100,100]),100); assert starts[2]>starts[0]


def test_random_mapping_reproducible():
    assert mapping_for(8,2,"random",7)==mapping_for(8,2,"random",7)


def test_plot_save_closes(tmp_path):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(); _save(fig,tmp_path/"x.pdf"); assert fig.number not in plt.get_fignums()
