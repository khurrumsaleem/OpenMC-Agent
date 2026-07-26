# OpenMC-Agent

把一句自然语言的反应堆建模需求，变成可审查、可校验、可运行的 OpenMC Python 模型。

`openmc-agent` 用 **Pydantic 结构化输出 + LangGraph 编排 + 可插拔渲染器** 把 LLM 限制在"只生成结构化数据"的安全角色里：LLM 不直接写代码、不直接调用 OpenMC，而是产出一个强类型的 `SimulationPlan`，再由本地渲染器将其转成 `model.py`，最后用 OpenMC 做 XML 导出、几何绘图和低粒子 smoke test。

> 设计原则：**LLM 只负责结构化建模决策，执行权完全在本地。** 缺失的物理数据会被放进 `requires_human_confirmation`，绝不伪造材料、密度或截面库。

系统 prompt 位于 `openmc_agent/prompts.py`，负责定义 Agent 能力边界、JSON-only 输出契约、缺失数据处理和 OpenMC 安全规范；`Input/case*.md` 只保留算例事实、默认假设、不确定项和审查目标。

---

## 核心特性

- **结构化 IR**：`SimulationPlan` / `ComplexModelSpec` 用 Pydantic v2 描述 pin-cell、组件、全堆、TRISO、燃料球等，校验失败即拒绝。
- **双工作流**：轻量 `SimulationSpec`（仅 pin-cell）与完整 `SimulationPlan`（复杂 IR + 能力评估 + 渲染）。
- **渲染能力分级**：`none → skeleton → exportable → runnable`，宁可产出"仅可审查的骨架"也不输出错误的可执行模型。
- **可插拔渲染器**：PinCell / RectAssembly / Triso / Core / Skeleton（兜底），按能力自动选优。
- **多模型适配**：内置智谱 GLM、DeepSeek 的 OpenAI 兼容 HTTP 客户端（含 SSE 流式、超时重试），其余走 aisuite。
- **检索增强**：本地 OpenMC Python API 内省 + few-shot 示例注入到生成 prompt。
- **可观测**：每次运行产出 `transcript.json`、`capability_report.json`、`TODO.md` 与 JSONL 运行记录。
- **Controlled investigation and structured repair**: Facts / Materials / Universes investigation and all five gate reviewers share a bounded two-attempt JSON/schema transaction; the business payload hash is fixed, and redundant tools stop once semantic evidence coverage is complete while uncovered targets remain fail-closed.
- **Phase 8C checkpoint/replay and closure**: accepted boundaries are persisted as sanitized, state-hashed snapshots in chronological order (`gate:facts` -> `patch:materials` -> `patch:universes` -> `gate:material_universe` -> `gate:placement` -> `gate:axial_geometry` -> `gate:assembled_plan`). Resume rejects corruption or fingerprint drift; downstream GateReplayBundle records a sanitized policy snapshot and accepted upstream chain. Preflight, recorded-review, and live-review support all five gates, and downstream replay invokes only the target gate's production preflight/evidence/reviewer path. MU preflight checks fuel universe `metadata.fuel_variant_id` against referenced fuel material `source_variant_id` before Placement can run. Placement reviewer coverage treats `complete_with_gaps` as complete only when every contract row and evidence ref is explicitly acknowledged. Campaign `--resume` skips only previously passed runs; failed runs are re-dispatched while retaining their run-dir checkpoints. `--stop-after-gate` now stops immediately after the target gate accepts (including accepted checkpoint resume) and classifies that milestone separately from full-plan canary success. Results expose coverage, blocking/rejected finding counts, terminal status, and sanitized diagnostics; raw prompts, reasoning, and provider responses are never replay artifacts. Full VERA4 canary remains milestone validation after target gate closure, not daily debugging.
- **Phase 8C Step 3E offline downstream qualification**: versioned Placement/Axial/Assembled bundles use an explicitly synthetic `offline_deterministic` accepted-upstream chain. `scripts/qualify_downstream_gates_offline.py` runs target-only production preflight and recorded-review in order and reports fixture fingerprints, coverage, blockers, rejected findings, and terminal status. Mutation corpus covers placement binding/location, axial domain/overlay, and assembled reference/renderer paths; this qualification does not replace MU acceptance or real live-review.
- **Phase 8C Step 3F campaign recovery qualification**: `CampaignRecoveryScenario` and `scripts/qualify_campaign_recovery_offline.py` exercise sanitized checkpoint/replay recovery with fake deterministic inputs only. The matrix covers clean five-gate reuse, input/policy/checkpoint/bundle/sensitive/upstream drift, timeout/schema/finding blockers, and dependency-graph downstream closure; results contain hashes, boundary reuse/invalidation, call counts, and stable issue codes only.
- **Phase 8C Step 3G downstream gate recovery + live-review orchestration**: downstream gate blocking→recovery cycle tests cover Placement retry_controller loop (resolved/no-progress/budget/stale) and Axial/Assembled re-replay recovery (block→mutate→accepted). `scripts/extract_downstream_replay_bundles.py` extracts `production_accepted` bundles from real campaign checkpoints; `scripts/run_downstream_live_review.py` orchestrates Placement→Axial→Assembled sequential review (live/recorded/preflight modes, early-break or continue-on-fail).
- **Phase 8C Placement retry targeting**: Placement preflight now treats control-state equality as required only for movable control inserts, preserves concrete missing universe IDs on deterministic issues, and propagates the full required ID set into dependency retry requests so Universes regeneration has explicit targets.
- **Phase 8C MU deterministic material checks**: MaterialsPatch validation, materials fragment qualification, and MU preflight now reject `atom_frac`/`weight_frac` compositions whose sums indicate unnormalized values (for example water `H1=2, O16=1` or alloy wt% values summing to `100`). Fraction bases must use normalized values (`0 < sum <= 1`); source percentages are divided by 100 during fragment qualification when the vector is unambiguous, and chemical ratios must declare `stoichiometric_ratio`.
- **Phase 8C shared insert-profile coverage**: inventory-driven universe requirements preserve all localized insert requirement IDs served by a shared radial profile. Fragmented Universes metadata records both the legacy singular ID and the complete ID set, and MU preflight can also recover coverage from `GeometryComponentInventory` for old checkpoints.

---

## 工作流

### SimulationPlan 工作流（`--plan`，默认当指定 `--plot`/`--smoke-test`/专家反馈时也启用）

```
receive_requirement
   → retrieve_openmc_docs      # 本地内省 OpenMC API，取相关符号/签名
   → select_few_shots           # 按结构特征+关键词挑 few-shot（抽象范式 + gold case）
   → generate_plan              # LLM 产出 SimulationPlan（带 normalization 容错）
   → validate_plan              # Pydantic 校验
   → repair_plan_format ─坏 JSON/坏 schema 重试──▶ validate_plan
   → generate_plan ──incremental 定点 patch 修复──▶ validate_plan
   → reflect_plan ──验证失败重试───────────────▶ validate_plan
   → assess_capability          # 本地重算 capability_report（覆盖 LLM 草稿）
   → semantic_audit             # P0-A: LLM 只读语义审查（warning_only / strict）
   → llm_repair_proposal        # P0-B: LLM 在 allowlist 内生成 RFC6902 补丁
   → run_supervisor             # P0-C: LLM 路由决策建议（advisory / controlled_route）
   → ask_expert                 # 可选：LangGraph interrupt/resume 专家反馈
   → render_plan_script         # choose_renderer 选渲染器 → model.py（或骨架）
   → execute_tools              # export_xml / 几何绘图 / smoke test
   → reflect_plan ──失败重试──▶ validate_plan
   → save_record
```

关键容错点：`generate_structured_output` 支持传入 `normalizer`，默认对 plan 启用 `normalize_capability_report`——LLM 若给出"非可执行却带具体 renderer"的矛盾 capability_report，会在 Pydantic 校验前被修正为 `supported_renderer="none"`，避免整个 plan 坍缩为 null。若模型返回坏 JSON 或 schema 不合格，Plan 工作流会先尝试格式修复；incremental assembled plan 校验失败时先以 validator issue 定位原 patch，LLM 仅能提交受 allowlist 约束的 RFC6902 patch edit，并在 clone 上经过 patch/assembly/full-plan validation 后才提交。无改善或重复候选才退回 targeted full-patch regeneration。

Plan closed-loop Phase 0 提供了可持久化 gate/stage 协议、预算和 JSON artifacts；`off` 始终保持既有工作流行为。

Plan closed-loop Phase 2：Placement Gate 将已接受的 Facts placement contract 与 universe/profile、intent、assembly/pin-map 和 core-layout 静态绑定做交叉审查。计数、坐标、profile/universe 引用由 Python 预检；独立 Placement Critic 只做证据约束的语义审查，且不会修改 patch。`advisory` 只记录结果，`controlled` 在 Facts accepted 后建立 placement-before-axial barrier。Placement revision 仅能在 issue-scoped path 上 clone→re-review→atomic commit；Facts/Universes 依赖只记录并阻断，尚未执行通用 dependency retry。最终 OpenMC root reachability 仍属于未来 Final Plan Gate。

### 渲染能力分级

| `renderability` | 含义 | 产物 |
|---|---|---|
| `none` | 无渲染器能处理 | 仅结构化 IR，供专家审查 |
| `skeleton` | 信息不全，产出审查骨架 | `model.py`（**不可执行**，`export_to_xml` 被注释）+ `TODO.md` |
| `exportable` | 可导出 XML，但不可运行 | `model.py` + XML 文件 |
| `runnable` | 完整模型 | `model.py` + XML + 可选 smoke test |

**3D assembly guard**：当需求包含轴向异质结构（axial layers、spacer grid、explicit z 范围、nozzle/plenum 等通用信号）但 plan 仍只是 2D assembly root 时，`openmc_agent/assembly3d_guard.py` 会在 plan validation 阶段（而非等到 renderer 抛错）就阻断导出，发出结构化 issue（`assembly3d.axial_layers_required` / `assembly3d.default_z_extent_for_axial_problem` / `assembly3d.spacer_grid_material_slab` / `assembly3d.pin_through_path_missing`），降级为 skeleton 或要求 human confirmation——避免产出 z=-1..1 的"形式可导出但物理错误"的伪 3D 模型。该 guard 只看通用词汇与 IR 形状，不含任何 benchmark 专用事实。

---

## 安装

需 Python ≥ 3.10 与 OpenMC（运行目标，不在 `pyproject.toml` 依赖里）。

```bash
conda create -n openmc-env python=3.10 -y
conda activate openmc-env
conda install -c conda-forge openmc        # OpenMC Python API 与可执行文件
pip install -e ".[dev]"                     # aisuite / httpx / langgraph / pydantic / pytest
```

### 云端开发环境

仓库提供可复现的云端环境配置：

- `environment.yml`：Conda/Mamba 环境，包含 Python 3.10、OpenMC 和 editable dev 安装。
- `Dockerfile`：基于 micromamba 的容器镜像，用于云端 runner 或本地 Docker。
- `.devcontainer/devcontainer.json`：Dev Container / Codespaces 配置，自动安装并运行轻量 smoke tests。

```bash
# Conda/Mamba runner
micromamba env create -f environment.yml
micromamba run -n openmc-env python -m pytest -q

# Docker runner
docker build -t openmc-agent .
```

密钥和截面库路径只通过云平台 secret / runtime env 注入，不提交到仓库。详细步骤见 `docs/cloud_environment.md`。

### 测试分层

Base Python 环境允许不安装 OpenMC；这类环境只跑 pure Python / no-OpenMC 测试：

```bash
make check-env
make test-no-openmc
```

OpenMC 只在 `openmc-env`、Docker 或 Dev Container 中保证。进入这些环境后再运行 OpenMC 相关检查：

```bash
make check-env-openmc
make test-openmc
```

完整验证用于已具备 OpenMC runtime 的开发/云端环境：

```bash
make test-all
```

`OPENMC_CROSS_SECTIONS` 仍只通过 runtime secret/env 注入；没有 OpenMC 的 CI/云端 runner 不应运行 `make test-openmc`。

---

## 配置

复制 `.env.example` 为 `.env` 并填入所用 provider 的 key：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `OPENMC_AGENT_MODEL` | `zhipu:glm-5.2` | `provider:model` 格式 |
| `ZHIPUAI_API_KEY` | — | 智谱 GLM |
| `DEEPSEEK_API_KEY` | — | DeepSeek |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | 走 aisuite 的 provider |
| `ZHIPUAI_TIMEOUT_SECONDS` / `_MAX_RETRIES` | 180 / 2 | provider 级超时与重试 |
| `OPENMC_AGENT_STREAM` | `1` | SSE 流式（慢模型建议开） |
| `OPENMC_AGENT_LLM_HEARTBEAT_SECONDS` | 10 | 心跳日志间隔 |

---

## 使用

CLI 入口：`scripts/run_inspect.sh`（封装）或 `python -m openmc_agent.inspect`。

```bash
# 从 Markdown 需求文件跑完整 plan 工作流（导出 + 绘图 + smoke test）
scripts/run_inspect.sh --md-file Input/case1.md --full

# 一句话需求
scripts/run_inspect.sh --requirement "建立一个 UO2 pin-cell 临界计算" --full --text

# 换模型 / 指定复杂组件用例
scripts/run_inspect.sh --model deepseek:deepseek-chat --md-file Input/case2.md --full

# 直接用 Python 模块
python -m openmc_agent.inspect "建立一个 2x2 组件模型" --plan --plot --smoke-test --json
```

`run_inspect.sh` 默认使用紧凑终端视图：显示当前 graph node、LLM 心跳、semantic audit / repair proposer / run supervisor 状态、报错和最终摘要，而不会回显整个输入或 SimulationPlan。完整 `transcript.json`、节点/报错日志 `cli.log`、以及模块 artifact 会保存在 `--output-dir`。需要在终端展开全部内容时使用 `--text`；需要 JSON stdout 时使用 `--raw-output`。

常用参数：`--plan`（强制 plan 工作流）、`--plot` / `--smoke-test` / `--full`、`--output-dir`、`--compact` / `--json` / `--text`（输出格式）、`--enable-semantic-audit`、`--enable-llm-repair`、`--enable-run-supervisor`、`--controlled-route`、`--expert-feedback`、`--interactive-feedback`、`--max-expert-rounds`、`--verbose`。

对于会输出较大 JSON patch 的 OpenAI-compatible 模型，可显式选择 JSON
对象模式、增加输出预算并关闭可选 reasoning 预算；这些参数默认均不改变原有
provider 行为：

```bash
conda run -n openmc-env python -u -m openmc_agent.inspect \
  --plan --verbose --md-file Input/case2.md --model ds:deepseek-v4-flash \
  --output-dir data/runs/case2_json \
  --patch-output-mode json_object --patch-max-tokens 12000 \
  --patch-reasoning-effort none \
  --plan-loop-mode advisory --plan-reviewer-output-mode json_object \
  --plan-reviewer-max-tokens 12000 --plan-reviewer-reasoning-effort none
```

`--patch-*` 作用于 Facts Proposer 与各 patch proposer；`--plan-reviewer-*`
可单独覆盖 Facts/Placement Critic。真实 provider 不支持 JSON mode 时会记录
fallback 实际模式；不可解析输出会保留原始 source requirement，并只从第一个
缺失或无效 patch 继续增量恢复，不会把诊断拼进后续 patch prompt 或调用
monolithic fallback。

默认 CLI 使用 `controlled` plan-loop，并启用当前可执行的 Facts、
Material–Universe 与 Placement gates；`--patch-output-mode` 仍为 `auto`。
因此普通 `inspect` 调用会自动进入增量建模路径。传入
`--plan-loop-mode off` 可恢复 legacy planning。Axial 与 Assembled Plan gates
尚未有可执行 controlled barrier，故不会被默认伪装为启用。

对包含多组件、轴向层和局部插入件的真实模型，使用 `controlled` gates；不要把
`advisory` 当作修复模式。受阻的 Facts Gate 不会被 Graph 的普通 patch retry
绕过：必须先重新获得 accepted Facts 才会生成下游。

```bash
conda run --no-capture-output -n openmc-env python -u -m openmc_agent.inspect \
  --plan --verbose --md-file Input/case2.md --model ds:deepseek-v4-flash \
  --output-dir data/runs/case2_controlled \
  --patch-output-mode json_object --patch-max-tokens 12000 \
  --patch-reasoning-effort none --max-plan-additional-llm-calls 20
```

这启用 Facts 与 Placement Gate；Material–Universe、Axial 与 Final Plan gates
仍不属于该命令的 controlled 范围。

交互式专家反馈：

```bash
python -m openmc_agent.inspect --md-file Input/case2.md --plan --interactive-feedback --max-expert-rounds 2
```

当 capability report 或 IR 中存在阻塞性人工确认项时，LangGraph 会通过 `interrupt` 暂停，CLI 展示问题并用 `Command(resume=...)` 把专家反馈写回图状态，然后重新生成/修复 `SimulationPlan`。

---

## Makefile 快速入门

所有命令默认用 `conda run -n openmc-env python` 执行（`aisuite` 在该环境），可通过 `PYTHON=` 覆盖。

### 单文件建模（真实 LLM）

```bash
# 最常用：跑 VERA3 3A（默认 deepseek）
make model INPUT=Input/VERA3_problem.md ALLOW_REAL_LLM=1

# 切换 variant / 堆型 / 输入文件
make model INPUT=Input/VERA3_problem.md VARIANT=3B ALLOW_REAL_LLM=1
make model INPUT=Input/VERA2_problem.md BENCHMARK=VERA2 VARIANT=2A ALLOW_REAL_LLM=1

# 切换 LLM
make model INPUT=Input/VERA3_problem.md MODEL=glm:glm-4-plus ALLOW_REAL_LLM=1
make model INPUT=Input/VERA3_problem.md MODEL=ds:deepseek-v4-flash ALLOW_REAL_LLM=1

# 不调 LLM，只看 feature detection（秒级，不花钱）
make model-dry INPUT=Input/VERA3_problem.md

# 带 OpenMC smoke test（输出 keff）
make model INPUT=Input/VERA3_problem.md ALLOW_REAL_LLM=1 SMOKE=1
```

**切换到 DS（SenseNova）模型**：

```bash
# 1. 设置 API key
export SENSENOVA_API_KEY="sk-xxx"

# 2. 用 ds: 前缀指定模型
make model INPUT=Input/VERA3_problem.md MODEL=ds:deepseek-v4-flash ALLOW_REAL_LLM=1

# 或者直接用 python
conda run --no-capture-output -n openmc-env python scripts/run_model.py \
    --input Input/VERA3_problem.md --variant 3A \
    --model ds:deepseek-v4-flash --allow-real-llm
```

> **提示**：`deepseek-v4-flash` 默认开启思考模式，CoT 会占用输出 token 预算，可能导致 JSON
> 被截断。本仓库默认对 `ds:` 注入 `reasoning_effort=low` 以抑制思考；如需调整可设环境变量
> `SENSENOVA_REASONING_EFFORT`（`none`/`low`/`medium`/`high`，留空则用 provider 默认）。
> 生成 patch 时不主动设 `max_tokens`——各 provider 的默认上限（如 DeepSeek ~8192）比任何
> 安全的统一 cap 都大，主动压低反而会截断多组件大 patch；参考预算见
> `openmc_agent/plan_builder/llm_adapter.py:PATCH_MAX_TOKENS`（可显式传给 `generate_patch`）。

可用 provider 前缀：`deepseek:`（DeepSeek 官方）、`ds:`（SenseNova 托管）、`zhipu:`（智谱）、`fake`（不调 LLM）。

运行时进度（`[node:...]`、`[llm] ...`）默认输出到 stderr。通过 `LOG_LEVEL` 控制：

```bash
make model ... LOG_LEVEL=WARNING   # 静默进度消息
make model ... LOG_LEVEL=DEBUG     # 更详细诊断
```

等价的 `--log-level` / `--quiet` 参数也可直接传给 `scripts/run_model.py`。
环境变量 `OPENMC_AGENT_LOG_LEVEL` 作为全局兜底。

输出写入 `data/runs/<BENCHMARK>_<VARIANT>/`，包含 `simulation_plan.json`、`model.py`、`incremental/material_composition_report.json` 和 traces。

### 回归 benchmark（evaluation cases 清单）

```bash
make benchmark-fake              # 快速 fake（秒级，不用 LLM/OpenMC）
make benchmark-real              # 真实 LLM，plan-only
make benchmark-save-baseline     # 把当前 fake report 固定为 curated baseline fixture
make benchmark-check             # 跑 + 对比 baseline + regression gate（一键验证）
```

`benchmark-check` 使用 fake model 重跑 `tests/fixtures/evaluation_cases.json`，并与 `tests/fixtures/workflow_baseline/evaluation_report.json` 比较；当 `pass_rate` / `plan_schema_success_rate` / `artifact_completeness_rate` 下降或出现新失败 case 时 exit 非 0，适合 PR gate。真实 LLM 验证使用 `make benchmark-real MODEL=...` 单独运行。

### 报告 diff（手动比较任意两个 report）

```bash
make diff-workflow-reports BASE_REPORT=path/a.json HEAD_REPORT=path/b.json
make gate-workflow-regression BASE_REPORT=... HEAD_REPORT=...
```

### 可覆盖的 Makefile 变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `INPUT` | `Input/VERA3_problem.md` | 输入 `.md` / `.txt` / `.json` 文件 |
| `VARIANT` | `3A` | 变体（3A / 3B / 2A …） |
| `BENCHMARK` | `VERA3` | 堆型标识 |
| `MODEL` | `deepseek:deepseek-chat` | LLM 模型（`provider:model` 格式） |
| `ALLOW_REAL_LLM` | （空 = 不允许） | 设 `=1` 才允许真实 LLM 调用 |
| `SMOKE` | （空 = 不跑） | 设 `=1` 跑 OpenMC smoke test |
| `REF_POLICY` | `off` | reference patch 策略（`off` = 纯 LLM，`reference_only_for_structural` = 参考优先） |
| `MAT_POLICY` | `apply_alloy_library` | 材料成分策略 |
| `OUT` | `data/runs/<BENCHMARK>_<VARIANT>` | 输出目录 |
| `CASES` | `tests/fixtures/evaluation_cases.json` | benchmark 用例清单 |
| `PYTHON` | `conda run -n openmc-env python` | Python 解释器 |

---

## LLM 智能化闭环（P0-A / P0-B / P0-C）

在确定性校验之后，三个 LLM 环节依次运行，形成"审查 → 修复 → 决策"的闭环：

| 环节 | 节点 | 能做什么 | 不能做什么 |
|---|---|---|---|
| **P0-A 语义审查** | `semantic_audit` | 只读检查 plan 语义一致性（轴向、材料、几何、边界） | 不修改 plan |
| **P0-B 补丁修复** | `llm_repair_proposal` | 在 path allowlist 内生成 RFC6902 补丁 | 不触碰材料密度、核数据、loading map |
| **P0-C 路由监督** | `run_supervisor` | 从 Python 计算的 `allowed_actions` 中选一个 | 不直接执行工具、不生成代码 |

### 安全机制

- **Python 先算 allowed actions**，LLM 只能从中选择
- **Python 可以 veto** LLM 的决策（13 种 veto code）
- **deterministic fallback** 在 LLM 不可用时自动接管
- **loop detection** 防止相同状态重复动作
- **retry budget** 限制每个 patch 的重试次数
- **protected paths** 阻止修改材料密度、核数据路径、benchmark 常数

### 三种模式

| 模式 | 行为 |
|---|---|
| `off` | 完全关闭 supervisor |
| `advisory`（默认） | 运行 supervisor，记录决策到 trace/artifact，**不改变真实路由** |
| `controlled_route` | supervisor 决策经 Python 验证后可影响路由，映射到已有安全节点 |

### 通过 `run_inspect.sh` 运行

```bash
# 单文件建模（默认开启全部 LLM 智能化，advisory 模式）
scripts/run_inspect.sh --md-file Input/VERA3_problem.md --state 3A --model deepseek:deepseek-chat --full

# Workflow benchmark（默认全部开启，6 个 case）
scripts/run_inspect.sh --benchmark --model deepseek:deepseek-chat --max-cases 6

# Benchmark + controlled-route（supervisor 决策影响真实路由）
scripts/run_inspect.sh --benchmark --model deepseek:deepseek-chat --controlled-route --max-cases 6

# 关闭某个环节
scripts/run_inspect.sh --benchmark --model deepseek:deepseek-chat --disable-supervisor
scripts/run_inspect.sh --benchmark --model deepseek:deepseek-chat --disable-audit --disable-repair
```

### 通过 Makefile 运行

```bash
# 单文件建模
make model INPUT=Input/VERA3_problem.md VARIANT=3A ALLOW_REAL_LLM=1

# Workflow benchmark（fake，不调用 LLM/OpenMC）
make benchmark-fake

# Workflow benchmark（真实 LLM）
make benchmark-real

# 跑 fake + 对比 curated baseline + regression gate（一键验证）
make benchmark-check
```

### 输出检查

运行完成后，查看 `benchmark_summary.md` 中的 **## Run Supervisor** 部分：

```
- completion rate: 100.0%
- action accuracy: 100.0%
- veto rate: 0.0%
- fallback rate: 0.0%
- human escalation accuracy: 100.0%
```

---

`Input/case2.md` 是一个 15x15 PWR 组件用例：默认组件包含燃料棒和导向管，另有一个候选 burnable poison universe，但默认不插入 lattice。完整流程会让 LLM 生成 `SimulationPlan`，本地 `RectAssemblyRenderer` 再做可达性分析，只渲染默认 lattice 实际使用的材料、cell、universe。

```bash
scripts/run_inspect.sh \
  --model deepseek:deepseek-chat \
  --md-file Input/case2.md \
  --full \
  --text
```

成功时，摘要应显示生成的是可执行 `model.py`，而不是 `Status: NOT EXECUTABLE` 的 skeleton：

```text
Generated model.py
Exported XML files: materials.xml, geometry.xml, settings.xml, tallies.xml, plots.xml
```

这个例子覆盖了几个组件建模中的常见坑：

- **组件 root 自动重建**：渲染器使用 `AssemblySpec.lattice_id` 生成 OpenMC root cell，LLM 输出中未插入的 root universe 不会阻塞默认模型。
- **候选 BP 不阻塞默认组件**：未插入 lattice 的 `burnable_poison_universe` 可以保留在 IR 中；其缺失的硼硅酸盐玻璃密度/成分只进入 warning 和 `TODO.md`。
- **UO2 富集材料安全渲染**：若 LLM 给出 `U235/U238` 的 weight percent 和 `O16` 的 atom ratio，渲染器会用 `chemical_formula="UO2"` + `enrichment_percent` 生成 OpenMC 接受的材料卡，避免 `Cannot mix atom and weight percents`。
- **几何边界兼容**：`rectangular_prism` 可作为复合 region 使用，并转换成当前 OpenMC API 的 `openmc.model.RectangularPrism`。

也可以离线跑固定 case2 回归验证，不调用远程模型：

```bash
conda activate openmc-env
python scripts/verify_case2_renderer.py
```

该脚本会检查默认组件不是 skeleton、候选 BP 材料没有进入 `model.py`、XML 可导出，并执行几何绘图/低粒子 smoke test 工具链。

---

## 输出

默认写入 `data/runs/<run>/`：

- `model.py` —— 渲染出的 OpenMC 模型（或不可执行的骨架）
- `transcript.json` —— 全流程结构化记录（需求 / IR / capability / 验证 / 工具结果）
- `capability_report.json`、`TODO.md` —— 骨架模式下的待办与缺口说明
- `materials.xml` / `geometry.xml` / `settings.xml`、`*.png` 截面图、`statepoint.*.h5`
- `inspect_runs.jsonl` —— 累积运行记录

---

## 项目结构

```
openmc_agent/
├── schemas.py            # Pydantic IR：Material/Geometry/ComplexModel/SimulationPlan/Capability
├── llm.py                # OpenAI 兼容客户端(智谱/DeepSeek) + 结构化输出 + normalization + repair
├── graph.py              # LangGraph 两条工作流（build_graph / build_plan_graph）
├── validator.py          # spec / plan / 生成脚本 校验
├── renderers/            # 可插拔渲染器：pin_cell / assembly / triso / core / skeleton + registry
├── renderer_authoring/   # 预留：agent 在线编写新渲染器（当前为安全受控的 stub，主流程未启用）
├── executor.py           # 早期直接渲染脚本（pin-cell + 复杂模型直写）
├── tools.py              # OpenMC 子进程工具：export_xml / 绘图 / smoke test
├── openmc_api.py         # 本地 OpenMC API 内省与文档检索
├── few_shots.py          # few-shot 选取（抽象范式 + gold case，堆型无关）
├── few_shot_cases.py     # gold case loader（slim IR / patch / 结构特征）
├── records.py            # JSONL 运行记录
└── inspect.py            # CLI 与可编程入口
tests/                    # 覆盖 schemas/llm/graph/renderers/executor/tools/validator 等
Input/                    # 示例建模需求（case1.md / case2.md）
```

## 协作与文档维护规则

仓库根目录维护两份 agent 规则文件：

- `AGENTS.md`：Codex 使用。
- `CLAUDE.md`：Claude 使用。

核心约定：

- **本仓库默认开启自动 commit/push**，显式覆盖全局 `~/.claude/CLAUDE.md` 中"不自动提交除非明确要求"的默认。
- 每次代码改动完成后，运行相关测试；能跑全量测试时优先运行 `conda run -n openmc-env python -m pytest -q`。
- 测试通过且确认改动范围无误后，自动 commit 并 push 当前分支。
- 自动提交时只 stage 本次任务相关文件，不把用户已有脏文件、临时脚本、PDF 或未确认输入资料混入提交。
- 每次重要代码或架构变更后，同步维护 `README.md` 和 `docs/project_technical_report.md`。
- `docs/project_technical_report.md` 是当前项目进度、架构状态、验证结果、风险边界和下一步建议的总入口。

## 默认检索策略

当前默认开启检索工具链：

```text
grep -> graph -> GraphRAG query planner -> GraphRAG -> plain RAG -> evidence ranking
```

默认策略会尽量用本地代码、测试、文档、知识图谱和 GraphRAG evidence 帮助 `reflect_plan` 修复结构问题，减少不必要的人工参与。对 cross section 路径、材料密度、composition、benchmark 常数、真实 loading map 等 fact gap，系统也会检索文档解释和配置上下文，但仍保留 human confirmation，不会自动编造缺失事实。

Knowledge Asset Runtime Loader：若设置 `OPENMC_AGENT_KNOWLEDGE_DIR`（或 inspect CLI 传 `--knowledge-dir`，或 `RetrievalPolicy.knowledge_graph_path`），orchestrator 会在 GraphRAG stage 自动加载 `data/knowledge` 中的持久化 graph nodes/edges 作为 `extra_nodes/extra_edges`；目录缺失或损坏只产生 warning，不影响 workflow。详见 `docs/knowledge_runtime_strategy.md`。

---

## 可插拔渲染器

每个渲染器继承 `BaseRenderer`，实现：

- `can_render(plan) -> RenderCapabilityReport`：不写文件，只声明能把这个 plan 做到哪一级（`none/skeleton/exportable/runnable`）。
- `render(plan, outdir) -> RenderResult`：写出 `model.py` 及 sidecar。

在 `renderers/registry.py:RENDERERS` 注册；`choose_renderer` 按 **runnable > exportable > skeleton > none** 选最高能力者，`SkeletonRenderer` 始终作为最后兜底（注册顺序须保持靠后）。新增渲染器只需实现接口并加入注册表。

---

## 测试

```bash
conda activate openmc-env
pytest                 # 或 python -m pytest
```

测试用 fake HTTP/LLM client，不依赖真实远程模型；涉及真实 OpenMC 执行的用例会在缺 OpenMC 时跳过。

---

## 局限

- 仅支持 `eigenvalue` 模式；复杂渲染器目前覆盖 pin-cell、矩形组件/全堆、TRISO/单球，其余落为 skeleton。
- `renderer_authoring`（agent 自主编写新渲染器）为预留接口，当前显式返回"未实现"，不会自动执行生成代码。
- LLM 偶发输出仍可能不合规，依赖 schema 校验 + normalization + repair 重试兜底；远程模型调用建议开启流式与重试。

### P0 workflow benchmark

The P0 workflow benchmark is a lightweight, report-generating evaluation entry point for the workflow trace contract. By default it uses the `fake` model, runs in `plan-only` mode, and does not call OpenMC or a real LLM.

```bash
make benchmark-fake                # fake model, no LLM/OpenMC
make benchmark-real                # real LLM (deepseek), plan-only
make benchmark-save-baseline       # save current fake report as curated baseline
make benchmark-check               # run fake + diff baseline + regression gate
```

`benchmark-check` runs the fake benchmark, compares against `tests/fixtures/workflow_baseline/evaluation_report.json`, and exits non-zero if `pass_rate` / `plan_schema_success_rate` / `artifact_completeness_rate` regress or new cases fail. Use `make benchmark-real MODEL=glm:glm-4-plus` for opt-in real LLM benchmarking.

The command writes `evaluation_report.json`, `benchmark_summary.md`, `traces/`, and `case_artifacts/` under the output directory.

### Single-model run (real LLM modeling on one input file)

```bash
# Defaults: VERA3 3A, deepseek, apply_alloy_library
make model INPUT=Input/VERA3_problem.md ALLOW_REAL_LLM=1

# Switch variant / benchmark / input file
make model INPUT=Input/VERA3_problem.md VARIANT=3B ALLOW_REAL_LLM=1
make model INPUT=Input/VERA2_problem.md VARIANT=2A BENCHMARK=VERA2 ALLOW_REAL_LLM=1

# Switch LLM model
make model INPUT=Input/VERA3_problem.md MODEL=glm:glm-4-plus ALLOW_REAL_LLM=1
make model INPUT=Input/VERA3_problem.md MODEL=ds:deepseek-v4-flash ALLOW_REAL_LLM=1

# Dry-run (resolve + feature detection, no LLM/OpenMC)
make model-dry INPUT=Input/VERA3_problem.md

# Run with OpenMC smoke test (keff)
make model INPUT=Input/VERA3_problem.md ALLOW_REAL_LLM=1 SMOKE=1
```

**Using the DS (SenseNova) model**: set `SENSENOVA_API_KEY` and use the `ds:` provider prefix:

```bash
export SENSENOVA_API_KEY="sk-xxx"
make model INPUT=Input/VERA3_problem.md MODEL=ds:deepseek-v4-flash ALLOW_REAL_LLM=1
```

Available provider prefixes: `deepseek:` (DeepSeek official), `ds:` (SenseNova-hosted), `zhipu:` (Zhipu), `fake` (no LLM call).

Output goes to `data/runs/<BENCHMARK>_<VARIANT>/` (overridable via `OUT=...`). Artifacts include `simulation_plan.json`, `model.py`, `incremental/material_composition_report.json`, and traces.

### P0-NEW-1: Controlled material composition policy

Structural alloys reduced to pure elements (Zircaloy-4 -> pure Zr, SS-304 -> pure Fe, Inconel-718 -> pure Ni) lose real absorption and bias keff high. The controlled alloy library (`openmc_agent/material_library.py`) provides nominal handbook compositions; the policy (`openmc_agent/material_policy.py`, default `apply_alloy_library`) substitutes them only for known structural alloys that the plan reduced to their base element. Fuel, water, helium, and pyrex are always preserved. Every substitution is recorded in a `material_composition_report.json` artifact.

Dry-run comparison (base Python, no OpenMC):

```bash
python scripts/compare_material_policies.py \
  --benchmark VERA3 --variant 3A \
  --input Input/VERA3_problem.md \
  --model fake --dry-run \
  --out data/evals/material_policy/VERA3_3A_dry
```

Real OpenMC smoke comparison (inside `openmc-env`):

```bash
python scripts/compare_material_policies.py \
  --benchmark VERA3 --variant 3A \
  --input Input/VERA3_problem.md \
  --model deepseek:deepseek-chat \
  --batches 5 --inactive 1 --particles 1000 \
  --allow-real-llm \
  --out data/evals/material_policy/VERA3_3A_alloy
```

The resulting `comparison_report.json` records `preserve_plan` keff, `apply_alloy_library` keff, and `delta_pcm`. See `docs/evaluation.md` for details and safety boundaries.
### Plan closed-loop Phase 1C

The incremental planner now reconciles a feature contract with Facts before
controlled downstream generation.  A persisted canonical scope selects exactly
one patch family (`pin_map` or `assembly_catalog` plus `core_layout`), and a
source-critical Facts preflight blocks scope/profile omissions before assembly.
Mass-derived overlay geometry also has an owner-aware material-density
readiness check.  These are deterministic safety checks, not Placement,
Material–Universe, Axial, or Final Plan review gates.
### Plan closed-loop Phase 3

Phase 3 adds a typed executable dependency-retry protocol for registered
Facts, Materials, Universes, canonical-task-plan, and placement-owner issues.
It uses clone validation, atomic owner commits, dependency-aware invalidation,
and resumable downstream rebuilds.  The protocol is disabled in `off`, records
only an execution plan in `advisory`, and never invokes a monolithic fallback
or an LLM Supervisor.

### Real-model structured-output recovery

The patch adapter now exposes provider-neutral controls for JSON response mode,
output-token budget, and optional reasoning effort.  Incremental
patch-generation failure preserves validated upstream envelopes and resumes at
the failed patch; its validator text remains structured execution metadata
rather than becoming part of the source requirement.  Large-lattice detection
accepts only explicit `NxM`/`N×M` notation, avoiding accidental interpretation
of hyphenated document identifiers as core dimensions.

### Facts Gate closure and controlled investigation

Facts revision is a bounded, source-backed closure transaction: a candidate is
committed only after rereview has no blocking findings. Partial repairs continue
with the remaining finding set for at most three rounds; duplicate candidates,
input-hash drift, invalid structured output, and unresolved human-only facts
remain fail-closed. The Facts completeness critic records downstream Materials,
Placement, Axial, and Universe gaps as warnings rather than treating them as
FactsPatch errors.

Controlled investigation first executes its mandatory baseline and recomputes
semantic coverage. If coverage is already complete, it skips the planner and
records `skipped_after_coverage_complete`; otherwise the existing two-attempt
structured planner transaction remains mandatory.
