"""
依据（公式 / 口径说明）

我们统计的是 “per GPU (per-rank) per iteration” 的发送流量，并将链路方向按你的口径拆分：
- 正向(Forward, clockwise): payload 主导
- 反向(Backward, counter-clockwise): ACK/控制流量主导，用 ack_ratio 近似
- PP: forward send 为 payload；backward 为 ACK 主导

(1) 参数量与梯度大小
给定模型参数量 P（个参数），梯度 dtype 字节数 b_g，则 DP 梯度 payload：
    DP_payload_bytes = P * b_g

(2) Ring allreduce 每卡发送量（payload）
对 group size = p 的 ring allreduce（经典通信体积公式），每 rank 发送 payload：
    ring_sent_bytes = 2 * (p - 1) / p * payload_bytes

(3) 方向拆分（符合你定义的“正向 payload、反向 ACK 主导”）
给定某个 collective 的 per-rank payload 发送量 sent_bytes：
    Forward_payload_bytes  = sent_bytes
    Backward_payload_bytes = 0
    Forward_ack_bytes      = overhead_bytes / 2
    Backward_ack_bytes     = ack_ratio * sent_bytes + overhead_bytes / 2

其中 overhead_bytes 用非常粗的“消息开销”估计：
    overhead_bytes = (2 * (p - 1) * ring_chunks) * msg_overhead_bytes

因此：
    Forward_total_bytes  = Forward_payload_bytes + Forward_ack_bytes
    Backward_total_bytes = Backward_payload_bytes + Backward_ack_bytes

(4) TP 通信（我之前用于生成表格的“简化模型”）
- 单次 TP 通信张量近似为 activation：
    act_bytes = microbatch * seq_len * hidden_size * b_act
- 假设每层 forward+backward 合计 tp_ops_per_layer 次集合通信（默认 4）
    TP_sent_bytes = n_layers * tp_ops_per_layer * ring_sent_bytes(act_bytes, TP)

(5) PP 点对点通信（均摊到每张卡）
- 每个边界发送 activation payload，均摊平均发送次数约为 (PP-1)/PP：
    avg_sends = (PP - 1) / PP
- forward:
    PP_forward_bytes = avg_sends * act_bytes
- backward（ACK 主导）:
    PP_backward_bytes = avg_sends * (ack_ratio * act_bytes)   (+ 少量 msg_overhead)

下面代码直接使用我之前那组示例数据（GiB / GPU / iter），画 4 组柱：
总流量、DP、TP、PP；每组两根柱（正向/反向），正反向各用一种颜色。
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data (GiB / GPU / iter)
# -----------------------------
groups = ["Total", "DP", "TP", "PP"]

total_fwd = 274.706
total_bwd = 4.292

dp_fwd = 257.1748046875
tp_fwd = 17.5
pp_fwd = 0.03076171875

fwd_sum = dp_fwd + tp_fwd + pp_fwd
dp_bwd = total_bwd * (dp_fwd / fwd_sum)
tp_bwd = total_bwd * (tp_fwd / fwd_sum)
pp_bwd = total_bwd * (pp_fwd / fwd_sum)

fwd = np.array([total_fwd, dp_fwd, tp_fwd, pp_fwd])
bwd = np.array([total_bwd, dp_bwd, tp_bwd, pp_bwd])

# -----------------------------
# Plot (log scale)
# -----------------------------
x = np.arange(len(groups))
w = 0.38

forward_color = "#1f77b4"   # blue
backward_color = "#ff7f0e"  # orange

plt.figure(figsize=(8, 4))

plt.bar(x - w/2, fwd, width=w, label="Dominant Direction", color=forward_color)
plt.bar(x + w/2, bwd, width=w, label="Opposite Direction", color=backward_color)

plt.xticks(x, groups)
plt.ylabel("Traffic (GiB / GPU / iter)")
plt.yscale("log")   # <<< 关键：对数坐标
plt.title("Dominant vs Opposite traffic (log scale)")

plt.legend()
plt.tight_layout()
plt.savefig("result.png")