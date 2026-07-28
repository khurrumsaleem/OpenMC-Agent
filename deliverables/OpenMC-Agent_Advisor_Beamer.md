# OpenMC-Agent：从自然语言建模到可信 OpenMC 执行

> Plan 与 Runtime 双智能闭环的受控 Agent 设计

- 汇报人：[姓名]
- [单位]
- 2026 年 7 月

<!-- 由对应 Beamer TeX 源转换；每个二级标题对应一页幻灯片。 -->

## 封面

技术路线、Agentic AI 机制、证据边界与下一阶段验收

## 汇报主线

1. 为什么不能让 LLM 直接运行核工程代码？
1. Plan 与 Runtime 如何形成两个受控智能闭环？
1. 哪些证据已经成立，哪些仍须保守？
> **核心信息**
> 核心命题：用结构化状态和确定性策略约束 LLM，使其决策可审查、可恢复、可验证。

## 问题：自然语言需求不是可执行核工程事实

**困难**

- 材料、几何、边界与源项耦合
- 输入经常缺失关键事实
- 代码可运行不代表物理可信

**目标**

- 生成可审查结构化计划
- 本地验证并决定是否执行
- 失败进入受控修复或人工确认

> **核心信息**
> 从“生成代码”转为“生成受约束建模决策”。

## 设计原则：不确定性必须走向正确的处理路径

1. LLM 提议；本地系统验证、提交与执行。
1. 缺失事实显式化，不由检索或 few-shot 补齐。
1. 局部失败优先局部修复；无改善时 fail-closed。
1. 每次接受都有状态、证据和预算记录。

## 三平面架构：统一控制面连接 Plan 与 Runtime

![三平面架构：统一控制面连接 Plan 与 Runtime](assets/01_three_planes.png)

## 职责边界：谁能做什么？

**LLM**

- 结构化计划
- 只读审计
- 受限 patch
- 动作建议

**本地系统**

- schema / policy
- renderer / OpenMC
- candidate 评估
- veto / budget

**人**

- 缺失材料事实
- 核数据路径
- 真实装载图
- 物理歧义

## Plan 工作流：生成不是最后一步

![Plan 工作流：生成不是最后一步](assets/02_plan_workflow.png)

## 结构化 IR：把模型输出转成可验证行动空间

- SimulationPlan / ComplexModelSpec：强类型材料、几何、lattice、axial、settings。
- normalizer + Pydantic：先修正机械矛盾，再拒绝不合格 schema。
- capability assessment：本地重算 renderer 能力，不相信 LLM 自报。
> **核心信息**
> none / skeleton / exportable / runnable 使“不知道”成为系统状态。

## Plan LLM 闭环：反思、修复与路由被拆开控制

![Plan LLM 闭环：反思、修复与路由被拆开控制](assets/03_plan_llm_loop.png)

## Plan 闭环的真正闭合点：唯一提交门

![Plan 闭环的真正闭合点：唯一提交门](assets/10_plan_contracts.png)

## P0-A / P0-B / P0-C 的 Agentic AI 含义

| 节点 | Agentic AI 功能 | 边界 |
| --- | --- | --- |
| semantic audit | 只读反思 / critic | 不修改计划 |
| repair proposal | 受限行动提议 | 仅 allowlist RFC6902 patch |
| run supervisor | 策略下的行动选择 | 只能选 allowed actions；Python veto |
> **核心信息**
> “智能”来自有上下文的判断；“可信”来自行动权限与提交条件被外部策略收紧。

## 案例一：语义不一致如何变成可审计的局部修复

1. validator 用稳定 issue 定位责任 patch；不直接全量重生。
1. LLM 在 allowlist 内提 RFC6902 patch。
1. clone 上执行 patch、assembly、full-plan validation。
1. 候选无改善、重复或触及 protected path：拒绝并转 human / stop。
[已实现机制] [具体结果按 case 证据陈述]

## 增量 Plan Builder：复杂模型不靠一次长输出

![增量 Plan Builder：复杂模型不靠一次长输出](assets/04_patch_gates.png)

## 检索增强：让模型获得上下文，但不获得造事实的权力

![检索增强：让模型获得上下文，但不获得造事实的权力](assets/05_retrieval_boundary.png)

## 检索在闭环中服务于判断，而非事实提交

![检索在闭环中服务于判断，而非事实提交](assets/11_retrieval_contracts.png)

## 状态管理：可恢复闭环不是“保存聊天记录”

![状态管理：可恢复闭环不是“保存聊天记录”](assets/06_state_replay.png)

## Gate 恢复：复用的是已接受边界，而不是历史对话

![Gate 恢复：复用的是已接受边界，而不是历史对话](assets/12_gate_state_machine.png)

## 案例二：Gate 恢复为什么可信

1. 保存已 accepted 的 Facts/MU 边界、issue 和 evidence hash。
1. 仅 replay 目标 Gate 的生产 preflight/evidence/reviewer 路径。
1. 输入、证据、库存和策略 fingerprint 全部相同才复用。
1. 漂移、损坏或敏感字段出现：拒绝复用，重新进入受控流程。
> **核心信息**
> memory 被实现为可验证状态，而不是可任意引用的历史对话。

## Runtime 确定性闭环：先分类，再修复

![Runtime 确定性闭环：先分类，再修复](assets/07_runtime_deterministic.png)

## 确定性优先：哪些错误能自动处理，哪些必须停下

**可进入安全修复**

- 已知 source/settings 引用错误
- 可证伪的冗余结构
- 已有 policy 明确所有权

**必须升级 / 停止**

- 材料组成、密度、温度
- 几何尺寸、坐标、核数据
- 真实装载或物理含义歧义

## Runtime LLM 闭环：LLM 生成候选，不拥有提交权

![Runtime LLM 闭环：LLM 生成候选，不拥有提交权](assets/08_runtime_llm_loop.png)

## Runtime：LLM 不是默认修复器

![Runtime：LLM 不是默认修复器](assets/13_runtime_decision_tree.png)

## 案例三：source 故障的受控恢复

1. OpenMC 产生真实 runtime failure；preflight / policy 将其归入 source/settings 类。
1. 确定性修复优先；若策略允许才调用 LLM 诊断和受限提案。
1. candidate 必须通过 schema、policy、isolated render、XML/debug 与真实 smoke。
1. 通过才提交和重执行；失败/无改善时有界停止。
[T4：真实 OpenMC 受控 source recovery] [不代表材料/几何物理保真度]

## Runtime Supervisor：智能循环必须有停止语义

![Runtime Supervisor：智能循环必须有停止语义](assets/09_runtime_supervisor.png)

## 有界自治：预算、veto、指纹和人机协同

| 控制 | 作用 |
| --- | --- |
| allowed actions + veto | LLM 不能提出政策外或不安全动作 |
| budget | 限制迭代、修复、重执行、LLM 调用和 candidate checks |
| fingerprint/no-progress | 阻断重复故障和无改善循环 |
| human confirmation | 事实缺口与高风险歧义显式升级 |
> **核心信息**
> fail-closed 是可信自治的终止语义，不是异常处理的缺省分支。

## 验证体系：T1--T6 防止证据越级

| 层级 | 能证明 | 不能证明 |
| --- | --- | --- |
| T1--T2 | schema/unit、生产图路由 | 真实 OpenMC |
| T3--T4 | 真实 OpenMC 基线/故障恢复 | LLM 稳定性、物理基准 |
| T5 | 真实 LLM pilot | 重复稳定性 |
| T6 | 真实 LLM N$≥$10 稳定性资格 | 物理 benchmark agreement |

## 当前 Runtime 证据：工程链路已有分层验证

- T1--T2：20 个注入 fault cases 通过。
- T3：真实 OpenMC 基线通过；T4：真实 source recovery 通过。
- T5：真实 LLM pilot 3/3；T6：10 次中 9 次 first-pass success，零 unsafe acceptance。
> **核心信息**
> 这些结果支持运行期工程闭环，不自动等价于模型物理保真度。

## Phase 8C：Plan Gate 闭环的阶段性成果

- 五 Gate、结构化 finding、checkpoint/replay 和 target-gate replay 已形成协议基础。
- 当前维护记录：非 OpenMC/LLM pytest 3715 passed；fake workflow benchmark 21/21。
- Placement 等下游真实 milestone 仍需按 production 证据继续验收。
[工程回归 / 离线证据] [下一阶段真实 milestone]

## 限制：可运行性与物理可信度必须分开

1. 低粒子 smoke test 主要验证基础设施。
1. LLM 生成模型与确定性 gold model 的 keff 仍有保真度差距。
1. renderer 覆盖、真实 provider 稳定性、复杂模型物理验收尚需扩展。
1. RAG、few-shot 与离线回放不是核工程事实来源。

## 下一步：工程闭环 + 物理闭环双重验收

1. 完成 Placement → Axial → Assembled 的真实 target-gate 验收。
1. 建立真实 workflow case、持久 trace 和评估 dashboard。
1. 推进 geometry/material fidelity、高粒子数计算和 benchmark 对照。

## 总结

1. **Plan 闭环**：结构化 IR、审计、受限 patch、Gate 与 replay。
1. **Runtime 闭环**：失败分类、确定性优先、LLM 候选、clone 评估与有界恢复。
1. **可信边界**：LLM 只提议；本地策略决定何时接受、执行、升级或停止。
> **核心信息**
> OpenMC-Agent 正在把“能生成”推进为“可验证、可恢复、可解释”。

## 致谢

**谢谢！**
欢迎讨论技术路线、验证边界与下一阶段验收设计

## 备份：技术栈与代表模块

| 模块族 | 职责 |
| --- | --- |
| schemas / validator | IR、跨结构校验与 stable issue |
| graph / run supervisor | Plan 编排与受控路由 |
| plan builder / closed loop | patch、assembler、Gate、checkpoint/replay |
| runtime repair / supervisor | RuntimeFailure、candidate evaluation、有界恢复 |
| trace / evaluation | artifact、真实性层级与 benchmark |

## 备份：五 Gate 输入与输出

| Gate | 审查对象 | 接受后 |
| --- | --- | --- |
| Facts | 来源、事实、确认项 | 下游 patch 生成 |
| Material--Universe | 材料与 universe binding | placement-owned patch |
| Placement | 位置、profile、assembly/core contract | 轴向处理 |
| Axial Geometry | layers、overlay、through-path | 最终组装 |
| Assembled Plan | 全局引用、renderer、完整计划 | 渲染执行候选 |

## 备份：T1--T6 术语

- **T1**: mock schema/unit。
- **T2**: production graph + injected ToolResult。
- **T3**: real OpenMC baseline。
- **T4**: real OpenMC fault and recovery。
- **T5**: real LLM end-to-end pilot。
- **T6**: repeated real-LLM stability qualification。

## 备份：Runtime Supervisor 默认预算

- 4 次 runtime iteration；3 次 committed repair / reexecution / deterministic attempt。
- 2 次 LLM diagnosis 与 proposal；1 次 transient retry；4 次 candidate OpenMC checks。
- 2 次 no-progress；提交后相同 fingerprint 默认 0 次容忍。

## 备份：术语表

- **IR**: 经 Pydantic 约束的中间表示。
- **Gate**: 具有 preflight/evidence/review 的阶段性接受边界。
- **checkpoint**: 已接受边界的脱敏 hash 快照。
- **replay**: 仅运行目标 Gate 生产路径的受限重放。
- **fail-closed**: 证据或策略不足时拒绝继续自动化。

## 备份：禁止性边界

- LLM 不直接写 renderer/XML，不直接调用 OpenMC。
- 不自动修改材料 composition/density、几何尺寸/坐标、核数据、真实装载图。
- 不把 fake client、注入工具或离线回放说成真实端到端验证。
