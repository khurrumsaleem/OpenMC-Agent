# OpenMC-Agent

把一句自然语言的反应堆建模需求，变成可审查、可校验、可运行的 OpenMC Python 模型。

`openmc-agent` 用 **Pydantic 结构化输出 + LangGraph 编排 + 可插拔渲染器** 把 LLM 限制在"只生成结构化数据"的安全角色里：LLM 不直接写代码、不直接调用 OpenMC，而是产出一个强类型的 `SimulationPlan`，再由本地渲染器将其转成 `model.py`，最后用 OpenMC 做 XML 导出、几何绘图和低粒子 smoke test。

> 设计原则：**LLM 只负责结构化建模决策，执行权完全在本地。** 缺失的物理数据会被放进 `requires_human_confirmation`，绝不伪造材料、密度或截面库。

---

## 核心特性

- **结构化 IR**：`SimulationPlan` / `ComplexModelSpec` 用 Pydantic v2 描述 pin-cell、组件、全堆、TRISO、燃料球等，校验失败即拒绝。
- **双工作流**：轻量 `SimulationSpec`（仅 pin-cell）与完整 `SimulationPlan`（复杂 IR + 能力评估 + 渲染）。
- **渲染能力分级**：`none → skeleton → exportable → runnable`，宁可产出"仅可审查的骨架"也不输出错误的可执行模型。
- **可插拔渲染器**：PinCell / RectAssembly / Triso / Core / Skeleton（兜底），按能力自动选优。
- **多模型适配**：内置智谱 GLM、DeepSeek 的 OpenAI 兼容 HTTP 客户端（含 SSE 流式、超时重试），其余走 aisuite。
- **检索增强**：本地 OpenMC Python API 内省 + few-shot 示例注入到生成 prompt。
- **可观测**：每次运行产出 `transcript.json`、`capability_report.json`、`TODO.md` 与 JSONL 运行记录。
- **通用堆型**：覆盖 PWR/BWR/VVER/HTGR/SFR/CANDU/MOX 等，材料/几何/栅格/边界均由输入文档驱动，不硬编码单一堆型规则。

---

## 快速开始

### 安装

需 Python ≥ 3.10 与 OpenMC（运行目标，不在 `pyproject.toml` 依赖里）。

```bash
conda create -n openmc-env python=3.10 -y
conda activate openmc-env
conda install -c conda-forge openmc        # OpenMC Python API 与可执行文件
pip install -e ".[dev]"                     # aisuite / httpx / langgraph / pydantic / pytest
```

#### 云端开发环境

- `environment.yml`：Conda/Mamba 环境（Python 3.10 + OpenMC + editable dev 安装）
- `Dockerfile`：基于 micromamba 的容器镜像
- `.devcontainer/devcontainer.json`：Dev Container / Codespaces 配置

```bash
micromamba env create -f environment.yml          # Conda/Mamba
docker build -t openmc-agent .                    # Docker
```

### 配置

复制 `.env.example` 为 `.env` 并填入所用 provider 的 key：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `OPENMC_AGENT_MODEL` | `zhipu:glm-5.2` | `provider:model` 格式 |
| `ZHIPUAI_API_KEY` | — | 智谱 GLM |
| `DEEPSEEK_API_KEY` | — | DeepSeek |
| `SENSENOVA_API_KEY` | — | SenseNova 托管（`ds:` 前缀） |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | 走 aisuite 的 provider |
| `OPENMC_AGENT_STREAM` | `1` | SSE 流式（慢模型建议开） |

可用 provider 前缀：`deepseek:`（DeepSeek 官方）、`ds:`（SenseNova 托管）、`zhipu:`（智谱）、`fake`（不调 LLM）。

### 最常用的建模命令

```bash
# 跑 VERA3 3A（默认 deepseek，含 OpenMC smoke test 输出 keff）
make model INPUT=Input/VERA3_problem.md ALLOW_REAL_LLM=1

# 切换 variant / 堆型 / 输入文件
make model INPUT=Input/VERA3_problem.md VARIANT=3B ALLOW_REAL_LLM=1
make model INPUT=Input/VERA2_problem.md VARIANT=2A BENCHMARK=VERA2 ALLOW_REAL_LLM=1

# 切换 LLM 模型
make model INPUT=Input/VERA3_problem.md MODEL=glm:glm-4-plus ALLOW_REAL_LLM=1
make model INPUT=Input/VERA3_problem.md MODEL=zhipu:glm-5.2 ALLOW_REAL_LLM=1
make model INPUT=Input/VERA3_problem.md MODEL=ds:deepseek-v4-flash ALLOW_REAL_LLM=1

# 不调 LLM，只看 feature detection（秒级，不花钱）
make model-dry INPUT=Input/VERA3_problem.md

# 带 OpenMC smoke test（输出 keff）
make model INPUT=Input/VERA3_problem.md ALLOW_REAL_LLM=1 SMOKE=1

# 紧凑终端视图（含绘图 + smoke）
make model INPUT=Input/VERA3_problem.md ALLOW_REAL_LLM=1 LOG_LEVEL=WARNING

# 开启交互式专家回环（需要时人工确认/补全缺失事实）
make model INPUT=Input/VERA3_problem.md ALLOW_REAL_LLM=1 INTERACTIVE=1
```

或者直接用 python（等价命令）：

```bash
conda run --no-capture-output -n openmc-env python scripts/run_model.py \
    --input Input/VERA3_problem.md --variant 3A \
    --model zhipu:glm-5.2 --allow-real-llm
```

> **提示**：`deepseek-v4-flash` 默认开启思考模式，CoT 会占用输出 token 预算，可能导致 JSON
> 被截断。本仓库默认对 `ds:` 注入 `reasoning_effort=low` 以抑制思考；如需调整可设环境变量
> `SENSENOVA_REASONING_EFFORT`（`none`/`low`/`medium`/`high`，留空则用 provider 默认）。

输出写入 `data/runs/<BENCHMARK>_<VARIANT>/`，包含 `simulation_plan.json`、`model.py`、`incremental/material_composition_report.json` 和 traces。

#### 可覆盖的 Makefile 变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `INPUT` | `Input/VERA3_problem.md` | 输入 `.md` / `.txt` / `.json` 文件 |
| `BENCHMARK` | `VERA3` | 堆型标识 |
| `VARIANT` | — | 工况变体（`3A`/`3B`/`2A` 等） |
| `MODEL` | `ds:deepseek-v4-flash` | `provider:model` |
| `ALLOW_REAL_LLM` | — | 设为 `1` 允许真实 LLM 调用 |
| `SMOKE` | — | 设为 `1` 跑 OpenMC smoke test（keff） |
| `INTERACTIVE` | — | 设为 `1` 开启专家需求回环 |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `WARNING` / `ERROR` |

---

## 工作流

```
receive_requirement
   → retrieve_openmc_docs      # 本地内省 OpenMC API，取相关符号/签名
   → select_few_shots           # 按结构特征+关键词挑 few-shot（抽象范式 + gold case）
   → generate_plan              # LLM 产出 SimulationPlan（增量 patch，带 normalization 容错）
   → validate_plan              # Pydantic 校验
   → assess_capability          # 本地重算 capability_report（覆盖 LLM 草稿）
   → render_plan_script         # choose_renderer 选渲染器 → model.py（或骨架）
   → execute_tools              # export_xml / 几何绘图 / smoke test
   → save_record
```

增量规划按 `facts → materials → universes → pin_map → axial_layers → axial_overlays → settings` 顺序生成 patch，每个 patch 独立校验与修复。若模型返回坏 JSON 或 schema 不合格，会先尝试格式修复；assembled plan 校验失败时以 validator issue 定位原 patch，LLM 仅能提交受 allowlist 约束的补丁，并在 clone 上经过校验后才提交。

### 渲染能力分级

| `renderability` | 含义 | 产物 |
|---|---|---|
| `none` | 无渲染器能处理 | 仅结构化 IR，供专家审查 |
| `skeleton` | 信息不全，产出审查骨架 | `model.py`（**不可执行**，`export_to_xml` 被注释）+ `TODO.md` |
| `exportable` | 可导出 XML，但不可运行 | `model.py` + XML 文件 |
| `runnable` | 完整模型 | `model.py` + XML + 可选 smoke test |

**3D assembly guard**：当需求包含轴向异质结构（axial layers、spacer grid、explicit z 范围、nozzle/plenum 等通用信号）但 plan 仍只是 2D assembly root 时，会在校验阶段就阻断导出，降级为 skeleton 或要求 human confirmation——避免产出"形式可导出但物理错误"的伪 3D 模型。

---

## 基准题演示（Demo）

| 算例 | 输入 | 建模内容 | 策略 |
|---|---|---|---|
| **VERA3 3A** | `Input/VERA3_problem.md` | 17×17 三维组件（空导向管、定位格架） | 增量 patch |
| **VERA3 3B** | `Input/VERA3_problem.md` | 17×17 三维组件（Pyrex 毒物棒、定位格架） | 增量 patch |
| **VERA2 2A** | `Input/VERA2_problem.md` | 17×17 三维组件（IFBA/BP/WABA、多种燃料富集度） | 增量 patch |
| **C5G7** | `Input/C5G7_problem.md` | 七群基准题 | monolithic 单次 plan |

### 运行方式

```bash
# VERA3 3A（推荐入门）
make model INPUT=Input/VERA3_problem.md ALLOW_REAL_LLM=1

# VERA3 3B（含 Pyrex 毒物棒）
make model INPUT=Input/VERA3_problem.md VARIANT=3B ALLOW_REAL_LLM=1

# VERA2 2A
make model INPUT=Input/VERA2_problem.md VARIANT=2A BENCHMARK=VERA2 ALLOW_REAL_LLM=1
```

### 如何读结果

- `data/runs/<case>/simulation_plan.json`：LLM 生成的结构化建模方案
- `data/runs/<case>/model.py`：渲染出的 OpenMC 模型
- `data/runs/<case>/materials.xml`、`geometry.xml`、`settings.xml`：导出的 XML
- `data/runs/<case>/plots/*.png`：OpenMC 几何校验图（材料/栅元切面）
- `data/runs/<case>/statepoint.*.h5`：输运结果（`k_combined` 即 keff）

`renderability` 分级见上文：`runnable` = 完整模型 + 可运行；`exportable` = 可导出 XML 但未运行；`skeleton` = 信息不全的审查骨架。结果清单如实标注状态，绝不伪造 keff。

---

## 回归 benchmark

```bash
make benchmark-fake              # 快速 fake（秒级，不用 LLM/OpenMC）
make benchmark-real              # 真实 LLM，plan-only
make benchmark-save-baseline     # 把当前 fake report 固定为 curated baseline fixture
make benchmark-check             # 跑 fake + 对比 baseline + regression gate
```

`benchmark-check` 跑 fake benchmark，对比 `tests/fixtures/workflow_baseline/evaluation_report.json`，若 `pass_rate` / `plan_schema_success_rate` / `artifact_completeness_rate` 回归或新增失败 case 则非零退出。

---

## 输出

默认写入 `data/runs/<run>/`：

- `model.py` —— 渲染出的 OpenMC 模型（或不可执行的骨架）
- `simulation_plan.json` —— 结构化建模方案（IR）
- `transcript.json` —— 全流程结构化记录（需求 / IR / capability / 验证 / 工具结果）
- `capability_report.json`、`TODO.md` —— 能力评估与骨架模式下的缺口说明
- `materials.xml` / `geometry.xml` / `settings.xml`、`*.png` 截面图、`statepoint.*.h5`
- `incremental/` —— 增量 patch 状态、材料组成报告、校验诊断
- `inspect_runs.jsonl` —— 累积运行记录

---

## 项目结构

```
openmc_agent/
├── schemas.py            # Pydantic IR：Material/Geometry/ComplexModel/SimulationPlan/Capability
├── llm.py                # OpenAI 兼容客户端(智谱/DeepSeek) + 结构化输出 + repair
├── graph.py              # LangGraph 工作流（build_plan_graph）
├── plan_builder/         # 增量 patch 生成、校验、修复、装配、Gate 闭环
├── renderers/            # 可插拔渲染器：pin_cell / assembly / triso / core / skeleton
├── executor.py           # 复杂模型渲染脚本生成
├── tools.py              # OpenMC 子进程工具：export_xml / 绘图 / smoke test
├── material_library.py   # 结构合金名义成分库（Zircaloy-4 / SS-304 / Inconel-718）
├── prompts.py            # 系统 prompt（能力边界 / JSON 契约 / 安全规范）
├── few_shots.py          # few-shot 选取（抽象范式 + gold case，堆型无关）
└── inspect.py            # CLI 与可编程入口
scripts/
├── run_model.py          # 单文件建模入口（最常用）
├── run_workflow_benchmark.py  # 回归 benchmark
└── compare_material_policies.py
tests/                    # 覆盖 schemas/llm/graph/renderers/plan_builder/validator 等
Input/                    # 示例建模需求（VERA2/3, C5G7, case1/2）
```

---

## 可插拔渲染器

每个渲染器继承 `BaseRenderer`，实现：

- `can_render(plan) -> RenderCapabilityReport`：不写文件，只声明能把这个 plan 做到哪一级（`none/skeleton/exportable/runnable`）。
- `render(plan, outdir) -> RenderResult`：写出 `model.py` 及 sidecar。

在 `renderers/registry.py:RENDERERS` 注册；`choose_renderer` 按 **runnable > exportable > skeleton > none** 选最高能力者，`SkeletonRenderer` 始终作为最后兜底。新增渲染器只需实现接口并加入注册表。

---

## 材料策略

结构合金被简化为纯元素（Zircaloy-4 → 纯 Zr，SS-304 → 纯 Fe，Inconel-718 → 纯 Ni）会丢失真实吸收、使 keff 偏高。默认策略 `apply_alloy_library` 只对已知结构合金用名义手册成分替换，燃料/水/氦气/Pyrex 一律保留。每次替换记录在 `material_composition_report.json`。

---

## 测试

```bash
conda activate openmc-env
python -m pytest                          # 全量
python -m pytest -m "not openmc"          # 不依赖 OpenMC 的用例（base Python 也可跑）
```

测试用 fake HTTP/LLM client，不依赖真实远程模型；涉及真实 OpenMC 执行的用例会在缺 OpenMC 时跳过。

---

## 局限

- 仅支持 `eigenvalue` 模式；复杂渲染器目前覆盖 pin-cell、矩形组件/全堆、TRISO/单球，其余落为 skeleton。
- `renderer_authoring`（agent 自主编写新渲染器）为预留接口，当前显式返回"未实现"。
- LLM 规划质量受模型能力影响：简单工况（如 VERA3 3A）多数模型能稳定生成 runnable 模型；复杂工况（含毒物棒、多燃料变体）可能需要更强的模型或多次重试。
