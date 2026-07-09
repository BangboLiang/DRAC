# OCS Asymmetry Modeling

这个仓库包含Llama 3 405B训练过程中通信建模的一系列脚本。每个脚本在前一个脚本的基础上增加新功能或改进。

## 脚本演进历史（按创建时间顺序）

### 1. `llama3_even_share.py` - 基础通信模型
**首次引入的功能：**
- 基础的通信时间模型
- 支持对称（symmetric）和非对称（asymmetric）链路类型
- TP、PP、DP三个通信域的固定带宽分配（均分：1/3, 1/3, 1/3）
- 支持多种集合通信算法：
  - AllGather: ring, recursive doubling
  - ReduceScatter: ring, recursive halving
  - AllReduce: ring, Rabenseifner, recursive doubling
  - P2P: 点对点通信
- 按步计算TP、PP、DP的总通信时间

### 2. `llama3_discrete_lane.py` - 离散带宽单元
**新增功能：**
- **离散带宽量化**（`--unit-bw-gbps`）：将带宽分配从连续模型改为离散单元（lanes）
- 带宽单元分配算法（基于Hamilton最大余数法）
- 对非对称链路，支持预留最小反向控制单元（`--asym-min-reverse-units`）
- 支持显示有效带宽利用率（Eff BW列）

### 3. `llama3_max_util.py` - 带宽分段优化
**新增功能：**
- **带宽重配置延迟**（`--reconfig-ms`）：模拟OCS在切换带宽分配时的延迟
- **Pre-planned带宽分段策略**：使用动态规划（DP）算法自动寻找最优带宽分段点
  - 对已知的通信节点序列进行分段
  - 每个分段内使用最小化通信延迟的带宽分配
  - 权衡重配置开销与通信增益
- 支持两种通信调度抽象：
  - `microbatch`：每个microbatch的TP/PP交错
  - `compact`：粗粒度的TP块、PP块、DP块
- 添加了trace输出功能（`--emit-comm-trace`）：
  - JSON格式
  - CSV格式
  - PNG可视化（如果有matplotlib）
- 对比三种策略：
  - **Preplanned**：DP分段优化
  - **One-shot**：整个step用一个带宽分配
  - **Static**：固定的均分带宽

### 4. `llama3_limit_degree.py` - 度数约束与链路分段
**新增功能：**
- **度数约束**（`--degree-k-total`）：限制每个节点同时活跃的双向peer连接数（k_tp + k_pp + k_dp ≤ K）
- **链路批处理延迟**（`--link-batch-ms`）：模拟在保持带宽分配不变的情况下，切换活跃peer连接的延迟
- **二层DP优化**：
  - 外层：带宽分段（BW segments）
  - 内层：每个带宽分段内部的链路分段（Link segments）
  - 每个链路分段选择最优的度数分配（k_tp, k_pp, k_dp）
- 更详细的分段报告：
  - BW segment级别的重配置开销、链路边界开销、内部retune开销
  - Link segment级别的度数分配和开销分解
- 添加了更多模型配置参数（`head_dim`, `kv_dim`, `ffn_hidden`等）用于更精确的payload计算

### 5. `llama3_refine_comm.py` - 集合通信算法配置增强
**新增功能：**
- **Collective算法profiles**（`--collective-profiles`）：支持预定义的算法组合
  - `mixed`：TP/PP用ring/asym，DP用hypercube（RH/RD）
  - `ring_asym`：全部用ring/asymmetric
  - `ring_sym`：全部用ring/symmetric
  - `hypercube`：TP/DP用recursive halving/doubling
  - `all`：运行所有profiles并生成组合对比图
- 支持多个profiles的组合PNG输出，时间轴对齐
- 优化的trace可视化：
  - 更精细的PP标签控制（`--comm-trace-pp-label-every`）
  - 短事件标记功能（`--comm-trace-plot-min-marker-ms`）
  - 可调整的时间缩放（`--comm-trace-plot-ms-per-inch`）
- 可选的static策略初始重配置（`--static-include-initial-reconfig`）

### 6. `llama3_modular.py` - 模块化重构
**新增功能：**
- **代码模块化**：将核心功能提取到`llama3_comm/`包中
  - `config.py`：配置类
  - `traffic.py`：通信量计算
  - `degree.py`：度数分配
  - `solvers.py`：DP求解器
  - `execution.py`：trace生成
  - `plotting.py`：可视化
- **Tree算法支持**（`tree` profile）：gather+broadcast / reduce+scatter树状集合通信
- **更精确的payload模型**：基于Megatron-style的layer-wise通信模式
  - 区分forward和backward pass
  - 每层的详细TP通信（QKV AllGather, AttnOut ReduceScatter, MLP AllGather, MLPOut ReduceScatter）
  - 每层的PP P2P传输
  - TP LayerNorm AllReduce
  - DP ZeRO-2 ReduceScatter（梯度，FP32）和AllGather（参数，BF16）
- 代码质量提升：更好的类型注解、文档字符串、模块化设计

### 7. `llama3_bucket_overlap.py` - 计算通信重叠模拟
**新增功能：**
- **DP bucketing**：将DP ReduceScatter分成多个bucket
  - 默认bucket大小：`max(40MB, 1MB × dp_world_size)`
  - 可自定义bucket大小（`--dp-bucket-bytes`）
- **计算时间估算**（`--gpu-tflops`, `--bwd-flop-fraction`）：
  - 基于6N参数×N token的训练FLOPs估算
  - 可配置backward占比（默认2/3）
- **简单的计算/通信重叠模型**：
  - 单流backlog模型
  - 计算DP tail时间（超出backward计算窗口的部分）
  - 报告DP payload时间、重叠时间、exposed tail时间
- **扩展的backward建模**：
  - 可配置最后一个microbatch的backward层数（`--bwd-last-mb-layers`）
  - 在backward过程中交错启动DP bucket通信
  - 每个bucket记录之前的计算时间（`gap_before_ms`）用于重叠计算
- 输出"有效总时间"（effective total）：用DP tail替代序列化DP时间

### `run_llama3_refine_grid.py` - 参数扫描工具
**功能：**
- **批量实验运行**：对多个参数组合进行网格搜索
- 支持的扫描参数：
  - `--unit-bw-values`：带宽单元大小（默认0-32）
  - `--reconfig-values`：重配置延迟（默认多个值）
  - `--degree-k-values`：度数约束（默认3-8）
  - `--bandwidth-values`：总带宽（默认50-800，步长25）
  - `--latency-values`：延迟（默认多个值）
- **并行执行**（`--workers`）：使用线程池并行运行实验
- **批次调度**（`--batch-size`）：分批执行以管理资源
- 生成运行清单（`run_manifest.jsonl`）：记录所有实验的参数和结果
- 支持dry-run模式查看命令

## DRAC Evaluation Simulation Framework

本仓库新增了一个独立的 `drac_eval/` 仿真框架，用于论文
`DRAC: Direction-aware Reconfigurable Asymmetric Connectivity for OCS-based AI cluster networks`
的 Evaluation 部分。这个框架不会改动现有 `llama3_*.py` 的行为，而是在项目根目录提供新的入口脚本。

### 目录结构

- `drac_eval/`
  - `config.py`：实验配置加载（JSON/YAML）
  - `traffic.py`：traffic demand 生成 / 加载
  - `allocation.py`：DRAC 与 baseline 的资源分配逻辑
  - `metrics.py`：指标计算
  - `runner.py`：实验执行器
  - `plotting.py`：结果绘图
- `run_drac_eval.py`：运行完整 evaluation
- `plot_drac_eval.py`：基于已有 CSV 和矩阵结果重画 figures
- `configs/drac_eval_smoke.json`：快速 smoke test 配置
- `configs/drac_eval_paper.json`：较大规模的 paper-style 配置
- `tests/test_drac_eval.py`：基础 sanity checks

### 依赖安装

推荐使用项目根目录执行：

```bash
python -m pip install -r requirements.txt
```

### 运行 Smoke Test

```bash
python run_drac_eval.py --config configs/drac_eval_smoke.json
```

如果默认结果目录正被其他程序占用，也可以覆盖输出目录：

```bash
python run_drac_eval.py --config configs/drac_eval_smoke.json --output-dir results/drac_eval_smoke_rerun
```

运行后结果默认保存在：

```text
results/drac_eval_smoke/
```

其中包括：

- `raw/results_raw.csv`
- `summary/results_summary.csv`
- `raw/matrices/*.json`
- `figures/*.png`

### 运行 Full Evaluation

```bash
python run_drac_eval.py --config configs/drac_eval_paper.json
```

默认输出目录：

```text
results/drac_eval_paper/
```

### 重新生成所有 Figures

```bash
python plot_drac_eval.py \
  --summary-csv results/drac_eval_paper/summary/results_summary.csv \
  --raw-csv results/drac_eval_paper/raw/results_raw.csv \
  --matrix-dir results/drac_eval_paper/raw/matrices \
  --out-dir results/drac_eval_paper/figures
```

### 当前支持的实验能力

- Traffic demand generation / loading
  - TP / DP / mixed / PP
  - cluster size sweep
  - asymmetry sweep
  - fixed random seed
- Network / resource model
  - fixed symmetric base network
  - OCS unit bandwidth
  - per-node port budget
  - total OCS link budget
  - symmetric baselines
  - direction-aware asymmetric DRAC allocation
- Metrics
  - estimated iteration communication time
  - bandwidth-demand matching error
  - symmetric capacity waste
  - network utilization
  - OCS port utilization
  - releasable / requested physical resources
- Outputs
  - raw CSV
  - summary CSV
  - reusable plotting scripts

### 基础测试

```bash
python -m unittest tests.test_drac_eval
```
