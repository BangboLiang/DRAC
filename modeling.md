# ACTINA 基础上的建模

原有的建模：

- 实际没有 overlap 的重构时间：不用改
- 分成三个 domain：不用改
- 以通信时间比例分配 domain 带宽：计算方式不再是 $\frac{w_d}{b_d}$

通信时间 $t=\frac{w_d}{b_d}$ 是一个流体模型，假设所有带宽都是有效的，然而不是；这与逻辑拓扑和集合通信算法有关，许多算法不能充分利用双向对称物理链路

原有假设每个节点的计算图基本一致（除了 PP），所以每个节点的每个 domain 带宽比例和大小均相同，落实到链路分配上应也是相同的

原有建模中没有明说同一个 transceiver 是否可以给一个链路的 Rx/Tx 分配不同的波长（也就是不同的带宽，即类似发 $3\lambda$，收 $1\lambda$，但其实现上收发是独立的电路，这也是下面假设的基础

## 集合通信算法（与逻辑拓扑）

环形：数据沿着环单向流动，1->2->3...每一步每个节点都从上个节点接收 payload，计算，然后发送到下个节点（每个节点同步并行进行）

hypercube：必须是 2 的整次方倍，同一时刻仅与某一维度的节点通信，如 6->0110，相邻的节点有 1110（14）、0010（2）、0100（4）、0111（7），一次 RS/AG 只通信这 4 次；NCCL 无此算法

binary tree：RS/AG/AR 都由一次向上一次向下的两个phase组成，区别只在每个子树的根节点做什么计算

## 对于分步的集合通信（DP/TP/PP）

1. 在每一步内，上下行链路很可能不是同时打满的，所以原有的 B 并不能粗暴地当作实际可用的带宽。引入一个 $\kappa$ 来表示顺着通信方向有效带宽的比例，真正的带宽其实是 $BW_{\text{eff}}=\kappa \cdot \frac{b_dB}{s_{out}}$，即该 domain 的带宽 / 连接到的节点数 \* 有效的带宽比例

   - 对于环形 collective+ 高度不对称的带宽（可以近似认为是带宽完全沿着通信方向分配），有 $\kappa=1$，$s_{out}=1$
   
   - 对于环形 collective+ 原有对称的带宽，基本仅有一半沿着通信方向，有 $\kappa=0.5$
   
   - 对于 hypercube（recursive doubling/halving），对称链路 + 对称数据交换，有 $\kappa=1$，$s_{out}=1或2$
   
   - 对于 binary tree，对称链路 + 两步不对称数据交换，有 $\kappa=0.5$，$s_{out}=2$（同时发 2 个、同时从 2 个收）
   
   - 对于 PP 的 p2p，由上层节点发送到下层节点，类似环形 collective
   
     > 注：对于复杂的调度，可能搞不赢
   
2. 上述的集合通信不止一步，每次发送的消息大小也不相同，与集合通信算法和节点数有关。设每个节点上的**初始数据量**为 $M$：

   - 对于 p 个节点上的 ring ReduceScatter，通信次数 $R=p-1$，每次消息大小 $m=M/p$
   
   - 对于ring AllGather，通信次数$R=p-1$，但是链路上每次都是完整的一个节点的数据，每次消息大小为$M$
   
     > 换句话说，RS之后剩下的数据量可以对应AG之前的初始数据量
   
   - 对于 ring allreduce，是先一次ReduceScatter再一次AllGather，所以 $R=2(p-1)$；每次消息大小$m=M/p$，因为AllGather环节的初始数据大小只有$M/p$，即在每个节点上reduce出来的分片
   
   - 对于 hypercube，通信次数 $R=\log_2p$ （必须是 2 的整数次幂），每次通信的消息量不定，但总的消息量为 $(p-1)M$（t 可以算总 tramsmission+propagation）；落实到具体 collective 同理
   
   - 对于 binary tree，AG/RS/AR 均为两个阶段，通信次数由树的层数决定，$R=2\lceil \log_2p \rceil$，单条链路上每次通信消息量均为 $M$（假设不 chunk 不 pipeline）
   
   - 对于 PP p2p，$R=1$，$m=M$
   
3. 有了带宽、通信次数和信息大小，再加上传播延迟 $\alpha$，可以得到总通信时间公式 $t=R(\alpha+\frac{m}{BW_{\text{eff}}})$

可以看到 hypercube 效率很高但带宽利用率低，只能 2 的整数次幂通信，需要的节点度数高；binary tree 差一些但仍然是 log_2p 级别，需要的节点度数为 3；ring 常常打满带宽，但是步数多。

### 来自NVIDIA的一点insight

arxiv:2507.04786v3 NCCL中实现了tree和ring两种variant。16个H200节点，在小消息的情况下tree占优势，1MB左右两者即打平，也就是说绝大部分情况下，ring都胜过tree，是带宽占主导部分。

## 一次 optimizer step 的计算/通信图

### 无 overlapping & bucketing 版

以 llama3 405B 为例，Megatron-style， all forward all backward 策略 PP，省略 LayerNorm，不开 activation recomputation

DP=128，PP=16，TP=8；总 batch size=2048，DP 分成 128 份，seq=8192，mbs=1seq，每步 16 个 mb；126 层四舍五入相当于每个 PP shard 分 8 层

每个节点（TP rank）的每层 activation 大小=seq 8192 * mb 16 * hidden 16384 * bf16 2Bytes / TP 8 = 32*16 MiB

每个 microbatch（一次 forward/backward）的每层 activation=32MiB

1. forward pass 1，b~g 反复执行 8 遍（每层一遍）

   1. PP P2P 从上一 stage recv 输入，rank activation=32MiB
   2. （计算 LayerNorm）
   3. TP（SP） AllGather linear_qkv，rank activation=32MiB
   4. （计算 QKV 分片、attention、out-proj）
   5. TP（SP） ReduceScatter ，重新分散 activation，256MiB->32MiB
   6. TP AllGather，rank activation=32MiB
   7. （MLP 计算，SwiGLU）
   8. TP ReduceScatter，rank activation=256MiB->32MiB
   9. PP P2P send 输出到下一 stage，rank activation=32MiB
2. forward pass 2~16，同上（总共有 16 个，根据 PP 策略不同顺序可能不同）
3. backward pass 1，b~g 反复执行 8 遍（每层一遍）

   1. PP P2P 从下一 stage recv grad，32MiB
   2. TP AllGather MLP FC2 grad，32MiB
   3. TP AllGather FC1 wgrad activation，32MiB
   4. TP ReduceScatter FC1 dgrad，256MiB->32MiB
   5. TP AllGather Proj grad，32MiB
   6. TP AllGather QKV wgrad activation，32MiB
   7. TP ReduceScatter QKV dgrad，256MiB->32MiB
   8. PP P2P send grad 到上一 stage，32MiB
4. backward pass 2~16，同上
5. TP AllReduce LayerNorm 等小参数，<20MiB（实际上常在 DP 做完后才发生，计算上忽略）
6. DP ZeRO-2

   - 当前层的 grad=QKV（302M）+proj（268M）+FC1（1.7B）+FC2（872M）=3.19B 参数量，8 层就是 25.52B
   - ReduceScatter 更新 grad，760MiB*8=6.08GiB->5.94MiB*8=47.52MiB
   - llama3 实际使用 fp32 grad sync，故实际数据量要翻倍
   - 实际情况下通常在本层 backward 中按 bucket overlap，边界是 bucket 大小而不是层

‍

为了减少计算量做如下简化：

- layer 由 8 改为 2，即 fwd 的两个 PP 中间有 4 对 AG/RS，bwd 有 6 对；用于展示 P2P->AG、RS->AG、RS->P2P

  - 对于 DP bucketing，可适当增加 layer 数创造 release 点（仅在最后一个 mb 的 bwd pass 中增加）
- mb 由 16 改为 2，用于展示 PP send fwd->PP recv bwd、PP send fwd->PP recv fwd、PP send bwd->PP recv fwd

### 基于 Megatron 的

1. **forward（以某个 PP stage 为视角；对 microbatch 0..7 重复一次）**

1. PP P2P recv 上一 stage 输入 activation（约 32MiB）

- 非交错 pipeline：【不 overlap】（recv 是阻塞式依赖）
- 交错/VPP pipeline：见“Pipeline 造成的 overlap（交错/VPP）”，这里的 recv 可以被“预取 + 延后 wait”

2. （计算LayerNorm / RMSNorm + 残差准备）

- 计算：逐元素 norm、scale、residual path 读写
- 【可 overlap：可以与上一条“异步预取的 PP recv”并行】（仅在交错/VPP 且 recv 采用异步预取时）

3. TP（SP）AllGather：为 QKV 线性层准备输入（32MiB -> 256MiB）

- 【可 overlap（条件）：只有在启用 TP comm overlap（TE userbuffer 那套）时，AllGather 才会与后续 GEMM 形成流水式重叠】
- 否则 【不 overlap】（AllGather 完成后才进入 GEMM）

4. （计算：QKV GEMM）

- 计算：[S*B, H] x [H, 3H]（按你的形状，S=8K/TP=1K，B=1）
- 【可 overlap：如果第 3 步是“可 overlap 的 AllGather”，这里就是 overlap 的 compute 侧】

5. 计算：Attention

- 计算（典型路径）：reshape/split heads + RoPE + QK^T + softmax + P*Vmaharmstone/btrfs
- 【可 overlap：如果交错/VPP 下已经提前发起了 PP send/recv，这里是最主要的计算窗口】

6. TP（SP）ReduceScatter：把 attention 输出重新回到 SP 分片态（256MiB -> 32MiB）

- 通信：ReduceScatter（把全量序列再切回每 rank 的序列分片）
- 【可 overlap（条件）：启用 TP comm overlap 时，ReduceScatter 可与其相邻的 GEMM（通常是“产生该张量/消费该张量”的线性层）形成流水重叠】
- 否则 【不 overlap】

7. TP（SP）AllGather：为 MLP FC1 输入准备（32MiB -> 256MiB）

- 同第 3 步的结论：默认不 overlap；启用 TP comm overlap 才能和 GEMM 重叠

8. 计算：MLP（SwiGLU）

- 计算：FC1 GEMM -> gate/up 激活（SwiGLU）-> FC2 GEMM
- 【可 overlap：如果第 7 步 AllGather 可 overlap，这里是 overlap 的 compute 侧】

9. TP（SP）ReduceScatter：MLP 输出回到 SP 分片态（256MiB -> 32MiB）

- 同第 6 步的结论

10. PP P2P send 输出到下一 stage（约 32MiB）

- 计算：无（纯通信）
- 非交错 pipeline：通常在该 microbatch 的 forward 末尾阻塞推进 【不 overlap】
- 交错/VPP pipeline：可以异步发起，把 wait 推迟到后面 【可 overlap：与其它 microbatch / 其它 model chunk 的计算窗口重叠】（见下文单列）

2. **forward：stage 2..16 同上（宏观上在流水里并行推进；这里不展开）**

---

3. **backward（以某个 PP stage 为视角；对 microbatch 7..0 反向重复一次）**

1. PP P2P recv 下一 stage 的 activation grad（约 32MiB）

- 非交错 pipeline：阻塞依赖 【不 overlap】
- 交错/VPP pipeline：可异步预取并延后 wait 【可 overlap：与同 stage 其它 microbatch 的计算重叠】

2. 计算：反传 MLP（从 FC2 -> SwiGLU -> FC1）

- 计算：dFC2（dgrad GEMM）、wgrad GEMM；SwiGLU 的逐元素反传；dFC1、wgrad 等

3. TP（SP）AllGather：为某些 wgrad GEMM 准备激活（32MiB -> 256MiB）

- 这里是“反传里真实存在的异步 overlap 点”之一：
- 【可 overlap：AllGather 可以先异步发起，然后先算 dgrad GEMM；等要做 wgrad GEMM 前再 wait】

4. TP（SP）ReduceScatter / AllReduce：把 dgrad 回到 SP 分片态（256MiB -> 32MiB）

- 这里是“反传里真实存在的异步 overlap 点”之二：
- 【可 overlap：ReduceScatter/AllReduce 可先异步发起，然后先做 wgrad GEMM；最后在需要使用已分片 dgrad 时再 wait】

5. 计算：反传 Attention（从 output proj -> softmax/QK^T -> QKV proj）

- 计算：dProj（dgrad/wgrad GEMM）、softmax backward、QK^T backward、dQ/dK/dV 等

6. TP（SP）AllGather：为 QKV / Proj 的 wgrad GEMM 准备激活（32MiB -> 256MiB）

- 【可 overlap：同第 3 步（先发起 gather，先算 dgrad GEMM，后 wait 再算 wgrad GEMM）】

7. TP（SP）ReduceScatter / AllReduce：QKV 的 dgrad（256MiB -> 32MiB）

- 【可 overlap：同第 4 步（先发起 RS/AR，先算 wgrad GEMM，后 wait）】

8. PP P2P send grad 到上一 stage（约 32MiB）

- 非交错：阻塞推进 【不 overlap】
- 交错/VPP：异步发起、延后 wait 【可 overlap：与其它 microbatch / 其它 model chunk 的计算重叠】

4. backward：stage 2..16 同上

---

5. “小参数 AllReduce（LayerNorm 等）”

- 更贴近实际的说法：这些参数（以及大参数）最终都会进入 DP/ZeRO 的 bucket 化 grad sync 体系里统一做 ReduceScatter/AllReduce；一般不需要把它们当成单独一条 “TP AllReduce” 流程
- 如果你只是做带宽/规模估算：可以继续把它们当成“额外很小的一点 DP 通信”，对总量影响很小

---

6. DP ZeRO-2（bucket 级别；重点是 overlap 发生在 bucket，而不是“按层”）
   按你给的每层参数量（单个 Transformer layer）：

- 每层参数数：QKV 302M + proj 268M + FC1 1.7B + FC2 872M = 3.19B params
- TP=8 后，每 rank 持有的参数（同层）：3.19B / 8 = 0.39875B params
  每层梯度通信量（按你原先的口径继续算）：
- 若按 fp16 梯度计：0.39875B * 2 bytes ≈ 0.7975GB ≈ 760MiB
- ZeRO-2 的 ReduceScatter（DP=128）后，每 rank 留下：760MiB / 128 ≈ 5.94MiB
- 若实际用 fp32 做 grad sync：把上面翻倍

  - RS 后：5.94MiB * 2 ≈ 11.88MiB（每层、每 rank）
    实际 overlap 点（两个方向）：

1. Grad ReduceScatter 与 backward compute 的 overlap

- 【可 overlap：当一个 bucket 的所有参数梯度在反传中“变为 ready”后，会立刻异步发起该 bucket 的 ReduceScatter；此时后续层/后续 bucket 的反传计算可以继续跑】
- 关键点：overlap 的粒度是 bucket，不是 layer；所以 bucket size 直接决定“能提前多久开始通信”

2. Param AllGather 与 forward compute 的 overlap（分布式优化器需要的参数同步）

- 【可 overlap：进入 forward 前/过程中，对下一批会用到的参数 bucket 先异步 AllGather；当前正在算的 layer 继续算】
- 通常还会有 “prefetch 下一 bucket” 的链式调度，因此会看到 bucket 之间的流水
  额外但很重要的现实约束（会影响你这份列表里 overlap 的“实际有效性”）：
- 在 PP>1 时，并不是所有 pipeline stage 都会做细粒度 bucketing；很多 stage 会退化成“更少 bucket / 甚至单 bucket”，于是 DP overlap 的收益主要集中在 pipeline 的关键 stage（常见是 pp rank 0），其它 stage 的 DP 通信往往不在关键路径上

---

Pipeline 造成的 overlap（单独列；不开 CP）
A) 非交错 1F1B（no VPP / 不做 model chunking）

- 结论：PP 的 overlap_p2p_comm 不可用
- 所以 PP recv/send 都是“算到哪就等到哪”，基本 【无 PP 计算-通信 overlap】
- 你还能依赖的 overlap 主要是：

  - TP：反传里 AllGather/RS(或 AR) 与 GEMM 的异步重叠（见 backward 第 3/4/6/7 步）
  - DP ZeRO-2：bucket RS 与后续反传计算重叠（见第 6 部分）
    B) 交错 / VPP（interleaving / 有 model chunks）
- 结论：PP 的 activation/grad P2P 可以异步发起，并把 wait 推迟
- forward 侧的典型 pattern：

  - 【可 overlap：在某个 microbatch 的 forward 结束时异步发起 send_next + irecv_prev（为下一个将要计算的 microbatch 预取输入）；随后立刻去算别的 microbatch/别的 chunk】
  - 【真正的同步点：在消费该输入 activation 之前才 wait】
- backward 侧同理：

  - 【可 overlap：在某个 microbatch 的 backward 结束时异步发起 send_prev + irecv_next（预取下一次 backward 需要的梯度）；随后去算别的 microbatch/别的 chunk】
  - 【真正的同步点：在消费该 grad 之前才 wait】

## DP求重构机会

跟ACTINA原文基本一致：输入通信图（顺序、操作、数据量等），总带宽，重构时间$r$，输出重构时机、DP/TP/PP domain带宽比例和最小通信时间。

$\text{OPT}_j=\min_i(\text{OPT}_{i-1}+r+cost_{i,j})$

其中，$cost_{i,j}$是对通信节点$[i,j]$中计算最优带宽分配，优化目标是**最低通信时间**（而非简单的与带宽成正比）。

## 不同参数组合下的趋势

- 以上方的一个 optimizer step 为一个单元，使用 microbatch=2、layers=2（主要是 PP 接 PP、TP 接 TP 的处理）构建序列式的 CommNode
- 没有 bucketing，没有 communication-computation overlapping，但是 link 和 bw reconfig 可以 overlap（且默认相等）=> 独立的 link reconfig 几乎总是变成 bw reconfig
- unit-bw：0 1 2 4 8 12 16 24 32
- reconfig（ms）：0.01 0.5 1.5 10 25 50 100
- degree：3 4 5 6 7 8（起码先让 tree 跑起来）
- bandwidth（GB/s）：50 100 200 300 400 600 800
- latency：0.1 0.25 1.0

基于不同的参数跑了计算脚本，得出：

- ring_asym vs ring_sym：91.6% win，当 reconfig_ms 过高时 fwd PP send 和 bwd PP recv 之间会多一次 reconfig，阈值大概是 10ms

  - 可能在复杂的 pipeline schedule 中会有问题，会出现 bwd-fwd、bwd-bwd、fwd-bwd 和 fwd-fwd 四种情况
  - 由于严重低估了 layer 的数量，这个 reconfig 实际上影响会小很多
- ring_asym vs hypercube：时间基本一致；传输数据量太小，延迟无法占主导（除非把 latency 加到 10）；degree 的 penalty 很大
- ring_asym vs tree：打爆

## 带宽分配粒度与最小反向带宽

对于传统光模块，可以依靠网口的breakout，如一个400G口breakout成4个100G口，每个100G口可以1️⃣同时给两个对端提供100G单向互联带宽，或2️⃣给一个对端提供100G双向互联带宽。而在ACTINA的光模块中，分配波长的最小粒度是32Gbps（？）

另外理论与实践均表明，NCCL over RoCEv2的流量并不全是单向的，如224MiB payload的unidirectional P2P就伴随着239MiB的正向流量和1.72MiB的反向流量。为了让控制消息正确回到发送端，可以在一定程度上给反向分配一个最小粒度的带宽。

## 扇出链路预算是有限的

每种通信算法需要跟不同数量的peer通信，不同域的peer通常各不相同，所以要么这个GPU扇出足够大，可以同时连接不同域所需要通信的几个peer，要么就需要在中途进行重构，将暂时用不上的链路与原本的peer断开，连接将要通信的peer。显然扇出是不可能无限大的，所以很可能需要在两个通信节点之间或者单个通信内部进行重构。虽然构建通信图的时候没有考虑计算，但是考虑到通信之间常有计算可以遮蔽链路重配置，而通信节点内部几乎一定会拖慢通信，所以算法会鼓励前者的重配置时机。

- ring只有两个固定的peer
- 普通tree有三个固定的peer，double binary tree可以有4个
- hypercube就需要$\log_2p$个peer，比如128个节点的巨大变态DP里就要与7个节点双向通信
- PP p2p更复杂一些，考虑到复杂的pipeline调度那peer及其方向就会很容易改变了

比如在TP=8 PP=128的情况下，均使用ring只需要总共4个peer，每个通信节点内也只有2个会活跃，只有前驱后继；而均使用hypercube的话，总共会连接3+7=10个peer，其中在DP 通信内部，会有7个peer在不同阶段活跃。如果同时并不能hold住7个peer，就不得不在通信期间进行重构。比如在hypercube中，设度数为4，就要先连接4个节点并通信，重构，再连接剩余的3个节点并通信，相当于两个batch，整个通信时间基本翻倍了。

> 这里的假设是重构是stop-the-world的，但是鉴于hypercube同时只与一个节点进行通信，未来也可以考虑把空闲链路的重构掩盖在其他链路通信时间内。

假设链路peer的重构时间与带宽域的重构时间一样，这样在每个上述DP确定的重构点之后，相当于有一次**零成本的链路重构**。另外，由于TP和DP的peer基本是不变的，且有一次免费链路重构，所以不需要担心分配给某个带宽域的peer数量对不上准确peer的问题（吧）。

带宽的域分配并不绑定链路分配，这是使用ACTINA光模块的假设，比如TP:DP:PP可以做到10:1:1，但是仍然可以把全部的链路都分给DP，而使用市面上的光模块显然做不到。至于在需要链路重构的时候要不要顺便进行一次带宽重构，可能需要多考虑。

求链路重构时机和分配的算法是上述带宽重构DP结果之上的细化。由于每次带宽重构的时候都会提供一次免费的链路重构，所以链路重构的DP范围可以**只限定在两次带宽重构的间隔（下称“区间”）里**。

> 如 `（带宽重构） - TP AllGather - TP ReduceScatter - PP Send - PP Recv - TP AllGather - （带宽重构）`，在这个区间内DP求链路重构的时机，比如可以在ReduceScatter和PP Send之间安排一次链路重构。

通信节点对应着peer请求流：

ring -> `{prev, next}`

- 表示 ring 一次需要同时和左右邻居通信，所以最少需要 `k >= 2`

pp p2p -> `{prev}` 或 `{next}`

- 对 PP，会从名字里推断是 send 还是 recv、上一跳还是下一跳
- asymmetric 时甚至把 `prev:send` / `prev:recv` 当成不同逻辑资源

tree -> `{parent, c1, c2}`（此处是普通二叉树）

- 表示树里父 + 两个子一起参与，最少需要 `k >= 3`

hypercube -> `{p1}, {p2}, ..., {pm}`

- 每一步只碰一个 partner，所以最少 `k >= 1`

之后，对于这个区间内所有子区间，假设这些子区间的通信节点之间不进行链路重构（只存在通信内部重构），枚举子区间内所有的度分配方案，并找出在这个度分配方案下，通信内重构次数最少的一个方案。这一步是为了链路重构DP做准备，因为通信间重构将完全由DP来划分，而度数分配方案在两次重构之间不变。

> 如对上面的那个区间，有子区间 `TP ReduceScatter - PP Send` 并有总度数为3，前者使用ring。一种分配方案是给TP度数1，给PP度数2，那么PP根本用不上这2条链路，TP则会陷入无尽的重构中；另一种方案则是给TP 2，给PP 1，那么就不需要内部重构了。显然这是一种更好的方案。

> 为了减少在区间内度数分配方案的搜索空间，做了一点小优化：只留下【有意义】的度数分配。如上面例子中TP只允许分到度数2，多了没意义用不上，少了则重构代价太高。或者如hypercube的DP=128需要与7个节点通信，给TP度数3和4都会触发一次节点内部重构，所以只枚举3，不枚举4。

最后就是在这个大区间内进行DP了，可以得出：

1. 考虑度数限制的该区间最小通信时间
2. 每一次通信间链路重构的时机
3. 采用的子区间序列（即在两次链路重构之间的度数分配方案）
