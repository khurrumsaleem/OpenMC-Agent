# OpenMC-Agent TODO Roadmap

维护日期：2026-07-28

当前基线：`main`，最近确认生产修复：MaterialsPatch source-declared coolant boron normalization；Pyrex annular poison / upper helium plenum deterministic validation；selected benchmark state fuel-variant scope；Facts reviewer normalization firewall；Facts source-note schema-boundary closure；Facts downstream-detail owner boundary；Facts required-coverage revision path closure；Facts JSON Patch/schema/schema-out metadata adapter hardening；Facts whole-source evidence reconciliation（chunk-local unsupported finding 降级）；localized-insert 详细几何 downstream-owner 路由；确定性 annular-insert universe oracle（pyrex_rod LLM 失败回退）
全量测试：非 OpenMC/非 LLM `3833 passed, 2 skipped, 393 deselected`；`compileall` 通过；fake workflow benchmark `21/21`；baseline diff 因 baseline 文件缺失跳过

本文档用于维护仓库级主线、已完成能力、近期任务和长期方向。  
它不是单次运行生成的 skeleton `TODO.md`，也不应记录一次性调试日志。

---

## 0. 当前阶段判断

项目已完成从"LLM 一次性生成大计划"到"分层规划、语义审查、受限修复、受控路由、确定性渲染和回归评估"的主要架构升级。Phase 8C 已将 VERA4 主线推进到五 Gate target-seed 通过、render/openmc-smoke 可复用 accepted seed 执行，并把最近暴露的 coolant boron ppm/质量份额单位问题和 Pyrex 环形毒物棒/上部氦气腔结构问题前移到生产写入/Universe validation 层做确定性约束。

当前重心：

```text
过去：逐个 Gate 真实 canary 中定位 wiring / reviewer / deterministic preflight bug
现在：用已通过 seed 做低成本生产路径回归，然后只跑一次从头真实 VERA4 里程碑验收
```

近期主线不再是继续增加新的 Agent 模块，而是：

1. 重新从当前生产路径跑 VERA3B 与 VERA4 base；旧五 Gate seed 已不再代表当前 Pyrex validator 合同，不能作为最终通过依据；
2. 确认生产 materials normalization 与 Pyrex profile validation 在真实 LLM path、fragmented path、retry path 都生效；
3. 只在上述低成本验证通过后，启动一次从头真实 LLM VERA3B / VERA4 base 建模里程碑；若 Facts split-review 再次阻塞，先抽取 normalized artifact 离线分类，不连续重跑；
4. 若从头真实 run 失败，导出脱敏 artifact/replay fixture 离线修复，禁止连续盲目重跑 canary；
5. 完整通过后再整理 deliverables、更新验收报告，并考虑将 reactor-neutral 模式抽象进 few-shot/memory 候选。

### 状态总览

| 能力 | 状态 |
|---|---|
| Incremental patch planning | ✅ 已完成并作为复杂模型默认路径 |
| LLM Semantic Plan Auditor | ✅ 已完成 |
| LLM Patch Repair Proposer | ✅ 已完成 |
| LLM Run Supervisor | ✅ 已完成 |
| Invalid incremental plan 诊断后重新生成 | ✅ 已完成 |
| Workflow trace / fake benchmark / evaluation metrics | ✅ 已完成基础闭环 |
| Composable lattice transformations | ✅ 已完成 |
| Nested component override / through-path preservation | ✅ 已完成 |
| Axial loading production materialization | ✅ 已完成 |
| Deterministic pin-map count repair oracle (P0-D4) | ✅ 已完成 |
| Deterministic shoulder-gap repair oracle (P0-D5) | ✅ 已完成 |
| Expert feedback semantics & skeleton blocker routing (P0-D5A) | ✅ 已完成 |
| Early lattice-loading validation & grid migration (P0-D5B) | ✅ 已完成 |
| Real-LLM DeepSeek VERA3A/3B zero-retry smoke | ✅ 已通过 |
| VERA3 geometry contract | ✅ 已建立 |
| VERA3 rendered XML point-probe tests | ✅ 已建立 |
| VERA3 plot provenance / annotated plots | ✅ 已建立 |
| VERA3 benchmark-accurate geometry | 🚧 未完成 |
| VERA3 benchmark-accurate materials | 🚧 未完成 |
| VERA3 keff / power acceptance | ⬜ 未开始正式验收 |
| VERA3 gold memory / few-shot 发布 | ⬜ 禁止在验收前进行 |
| VERA4 Facts/MU/Placement/Axial/Assembled target gates | ✅ 已通过 seed/target 里程碑 |
| VERA4 render/openmc-smoke seed path | ✅ 可从五 Gate seed 执行 |
| VERA4 production material-unit normalization | ✅ 已接入 MaterialsPatch 写入层 |
| VERA4 Pyrex annular poison / upper helium plenum validation | ✅ 已接入 UniversesPatch deterministic validation |
| VERA4 从头真实 LLM 建模验收 | 🚧 下一主线里程碑 |

---

## 1. 最近完成的关键里程碑

### 1.1 确定性结构修复闭环 ✅

P0-D1 到 P0-D5B 建立了三级确定性修复 oracle，在 LLM 介入之前自动修复已识别的结构缺陷：

```
validate_plan → structural issue detected → deterministic oracle → clone evaluation → accept/reject
     ↓ (if rejected)                                                                    ↓ (if accepted)
targeted LLM patch repair                                                     re-validate → proceed
     ↓ (if rejected)
fresh incremental regeneration
```

#### P0-D1：Validation-Driven Incremental Patch Repair ✅

- plan-level issue 进入 patch-relative RFC6902 repair request（稳定 fingerprint、issue ownership policy、protected-path denylist、clone-only assemble/full validation）；
- accepted repair 不消耗 graph retry；
- 重复/无改善候选立即停止并退回一次 targeted regeneration。

#### P0-D2 / P0-D3：Real-LLM Patch Repair Smoke ✅

- 修复 real-model validation repair 延迟构造 patch adapter；
- 优先使用 adapter JSON mode；
- schema-invalid proposal 写入 proposal/evaluation artifact；
- DeepSeek 真实两次局部 RFC6902 proposal 完成完整 clone evaluation。

#### P0-D4：Deterministic Pin-Map Repair Oracle ✅

- assembler 始终尊重 `PinMapPatch.default_universe_id`；
- 唯一等量 default 修复 oracle、clone-only preflight；
- 真实 VERA3 3A 直接恢复 `fuel_pin=264, guide_tube=24, instrument_tube=1`。

#### P0-D5：Early Structural Validation & Deterministic Shoulder-Gap Repair ✅

- validator/renderer 共享 `assembly3d_structural_issues()`；
- `component_profile_as_material_slab` 在 validate_plan 阶段直接产生；
- 确定性 multi-patch repair bundle（universes + axial_layers）+ clone-only acceptance + through-path preservation；
- graph repair pipeline 新增 component-profile oracle。

#### P0-D5A：Expert Feedback Semantics & Skeleton Blocker Routing ✅

- `capability_blockers.py` 确定性分类 structural/environment/human_fact/fidelity；
- `expert_feedback.py`：ExpertQuestionGroup/Decision/AssumptionAcknowledgement；
- structural blocker 优先于 human confirmation；
- assess 节点 axial materialization dry-run probe。

#### P0-D5B：Early Lattice-Loading Validation & Deterministic Grid Migration ✅

- 共享 `lattice_loading_validation.py`：`lattice_transform.replacement_universe_missing` 在 validate_plan 阶段直接发现；
- patch-level transformation cross-reference validation（replacement/source universe 存在性、cell-id-as-universe 误用、spacer_grid_transformation_misuse）；
- `grid_loading_repair.py`：证据层级诊断 + Strategy A-D（优先移除冗余 grid transformation，不创建 solid grid universe）；
- graph 新增 lattice-loading/grid migration oracle；
- 真实 VERA3B retry_count=0、grid_cell 引用消除、capability runnable。

### 1.2 LLM 智能闭环 ✅

#### P0-A：Semantic Plan Auditor ✅

- structured semantic findings；deterministic fallback；warning-only / strict evaluation 模式；trace 与 artifact；benchmark precision / recall / false-positive 指标。

#### P0-B：Patch Repair Proposer ✅

- RFC6902-style 受限 patch；issue-code path allowlist；protected path denylist；clone apply；before/after deterministic validation；accepted / rejected / unsafe 分类；proposal-only / validate-only / apply-if-safe 模式。

#### P0-C：Run Supervisor ✅

- Python 先计算 allowed actions；deterministic veto；advisory / controlled-route；retry budget；state fingerprint；no-progress / loop detection；deterministic fallback。

### 1.3 Incremental planner 稳定化 ✅

复杂模型默认采用 7-patch 分层生成 + 确定性组装，已具备局部重试、reference patch policy、failed patch 诊断、assembled plan validation 失败后诊断再生成。

### 1.4 VERA3 geometry IR 与 renderer 架构 ✅

支持 `replace_universe_family` / `coordinate_override` / `nested_component_override`，多 loading 组合，legacy 迁移，shoulder through-path 保留，materialization failure 阻止静默回退。

### 1.5 VERA3 验收工具基础 ✅

canonical geometry contract、3A/3B patch fixtures、plan-level acceptance、rendered `geometry.xml` point probes、geometry hash differentiation、XY/XZ plots、annotated plots、SHA256 provenance。

---

## 2. 重要纠正：旧里程碑不能继续作为验收结论

### 2.1 不再检查固定"轴向 12 层"

轴向层数由所有 component profile、finite insert、grid overlap 和 shoulder/nozzle 边界的并集决定。正确验收指标是 axial domain 连续、无 gap/overlap、每个 component profile 的 z coverage 正确、同一 z 段 loading/overlay 组合正确、rendered point probes 正确。

### 2.2 旧 keff 仅是历史 smoke 结果

旧记录 3A ~1.149、3B ~0.979 是在几何与材料仍存在明显近似时得到的 smoke 结果，不能作为 benchmark acceptance。在 fuel gap、plenum 半径、Pyrex upper gas、spacer-grid 质量守恒、nozzle mixture、variant-specific coolant、controlled alloy composition 和正式收敛设置完成前不得据此评价模型准确度。

### 2.3 "能导出 XML"不等于 benchmark-accurate

```text
schema-valid → renderable → XML-exportable → geometry-accepted → physics-material-accepted → numerically benchmark-accepted
```

只有最后三层全部通过，才可晋升为 VERA3 gold model。

---

## 3. P0 当前主线：VERA3 Gold Model

### P0-V0：重新验证最新 production renderer 🚧 立即执行

目标：确认最终 XML 已真实包含 axial transformations。

需要重新生成 `data/evals/vera3_geometry/3A` 和 `3B`，检查 point probes（fuel→UO₂、z=382→helium、clad→Zircaloy-4、3B Pyrex annulus→Pyrex、thimble→SS304、shoulder guide wall→Zircaloy-4），geometry hashes 3A/3B 差异，plot manifest 无错误，geometry debug 无 overlap/lost particle。

### P0-V1：燃料棒径向 gap 与 plenum 修正 ✅

活性燃料棒正确径向结构：`0–0.4096 UO₂ / 0.4096–0.418 helium gap / 0.418–0.475 Zircaloy-4 clad / r≥0.475 coolant`。

上部气腔：`0–0.418 helium / 0.418–0.475 clad / r≥0.475 coolant`，不得用 `0.4096 cm` 作为 plenum/clad 分界。

完成标准：production fixture、geometry contract、acceptance helper 一致；active fuel gap 和 plenum 有 rendered point probes；fuel/gap/clad volume 可计算。

### P0-V2：完成 3B Pyrex upper-gas axial profile ✅

Pyrex poison segment 15.761–376.441 cm；upper helium region 376.441–397.510 cm。需要新增 `pyrex_upper_plenum_inner_profile` / `pyrex_upper_plenum_loading`，重新切分轴向层（376.441–397.510 区间），在 386.267–390.133 cm 叠加 top spacer-grid overlay。

完成标准：Pyrex 不在 376.441 cm 后错误恢复为水导向管；16 Pyrex path 保留正确上部气腔；8 thimble path 坐标互斥；multi-loading + top grid overlay 同时正确。

### P0-V3：Spacer-grid 质量守恒几何 ✅

实现 mass/volume-conserving outer-frame overlay（`mass_conserving_outer_frame`）。端部格架 Inconel-718 1017 g / 3.866 cm；中间格架 Zircaloy-4 875 g / 3.810 cm；质量平均分配到 289 cells；grid material 只占 pitch cell 最外薄方框。

完成标准：grid material volume 与输入质量容差内一致；不再替换整个 moderator open region；through-path 保留。

### P0-V4：Nozzle / core plate 均匀化材料 ✅

正确 volume fractions：lower nozzle 27.92% SS304 / 72.08% coolant；upper nozzle 19.15% / 80.85%；core plate 50% / 50%。3A/3B 使用各自 coolant。

完成标准：不再用纯 SS304 slab 替代 mixture；mixture provenance 写入 material report；3A/3B mixture 不被错误复用。

### P0-V5：材料组成可信化 ✅ 已被 P0-FINAL 合并

VERA3 reference Table P3-3 的精确原子数密度已直接写入 fixtures（`atom_density_barn_cm`）。所有存根已被替换：UO₂ isotopics、Zircaloy-4（29 nuclides）、SS304（19 nuclides）、Inconel-718（21 nuclides）、Pyrex、helium、variant-specific coolant、pre-homogenized nozzle/core-plate。

### P0-V6/V7：几何 Gold + 数值 Acceptance — DEFERRED

已通过 transport smoke（3A Δ=0.025%, 3B Δ=0.53%），但正式 gold/numerical acceptance 推迟到用户提供更高统计要求后的 production 验证。

### P0-FINAL + FC-MVP：工程基线闭合 + 全堆 MVP ✅

- VERA3 engineering baseline + transport smoke passed
- Reusable 3D assembly universe（transmission boundaries）
- Full-core MVP: 3×3 core lattice + reflector + vacuum BC
- Full-core transport smoke: keff=0.99452±0.005, 0 lost particles, assembly tally 9 entries

---

## 4. P0：真实 LLM 与评估闭环

### 4.1 Real-LLM advisory baseline 🚧

VERA3A/3B DeepSeek zero-retry smoke 已通过。继续测试更多 case：clean exportable、failed patch、fact-gap、unsupported subsystem、axial/profile conflict、unsafe material repair proposal。

记录 semantic precision / false positive、repair accepted/rejected/unsafe、supervisor action accuracy、latency/token/cost。

### 4.2 Workflow benchmark regression gate ✅ 基础闭环完成

report-to-report diff、case status changes、new regressions、fixed cases 已支持。继续完善 unsafe repair count、supervisor unsafe action、fact-gap bypass、real LLM opt-in baseline trends。

### 4.3 Fact-gap regression suite 🚧

覆盖 density missing、composition missing、benchmark constants missing、loading map missing、nuclear data path missing、ambiguous axial bounds、conflicting source documents。

完成标准：retrieval/few-shot/auditor/repair/supervisor 均不能自动确认缺失科学事实。

---

## 5. P1：Unseen-model 智能能力

### P1-A：LLM Evidence Synthesizer ⬜

将 grep / graph / GraphRAG / RAG 证据压缩为结构化 brief（claims、locators、contradictions、unsafe-auto-fill flags、missing evidence）。先离线评估。

### P1-B：LLM Task Decomposer ⬜

对 unseen models 生成 required patch types、order、capabilities、known facts、fact gaps、risk tags、verification plan。与 deterministic feature detection 并行。

### P1-C：Retrieval ablation ⬜

对比 grep/graph/GraphRAG/RAG/ranker/synthesizer 各组件 off 时的 task success delta。

### P1-D：Supervisor-driven multi-round repair ⬜

audit → repair proposal → deterministic validation → targeted patch retry → reassemble → revalidate → re-audit → render/ask human/stop，第一阶段最多 1–2 轮。

---

## 6. P1：Renderer 与物理能力扩展

只有 VERA3 fidelity 主线稳定后推进。

- **6.1 HexAssemblyRenderer** ⬜ — OpenMC HexLattice、ring schema、skeleton → exportable。
- **6.2 Depletion / burnup boundary** ⬜ — IR、operator boundary、material evolution、先定义 capability。
- **6.3 Pebble-bed / stochastic geometry** ⬜ — 明确 unsupported boundary、skeleton 输出。
- **6.4 Boundary-condition rendered verification** 🚧 — radial/axial boundary、XML `boundary_type`、leakage anomaly。

---

## 7. P1/P2：VERA3 Memory 与 Few-shot

### 前置条件

只有满足以下条件才能发布：geometry gold accepted、material fidelity accepted、numerical acceptance documented、provenance complete、reference_policy=off reconstruction evaluated。

- **7.1 Benchmark-specific memory** ⬜ — `data/benchmark_knowledge/VERA3/` 结构。
- **7.2 Reactor-neutral few-shot** ⬜ — 从 gold model 匿名提取 component profiles、family replacement、through-path preservation、spacer overlay 等模式，不保留 VERA3 名称/坐标/常数/keff。
- **7.3 Accepted-run mining** ⬜ — 从 accepted traces 自动生成候选 exemplar，人工 review 后晋升。

---

## 8. P2：工程化与用户体验

- **8.1 CI 分层** ⬜ — fast unit / integration / OpenMC geometry / OpenMC transport / real LLM opt-in。
- **8.2 Artifact 目录统一** ⬜ — plan/incremental/retrieval/semantic_audit/repair/supervisor/render/verification/transport/evaluation。
- **8.3 用户可读 run summary** ⬜ — 当前状态、renderability、blockers、unresolved facts、approximations、supervisor decision、required action、artifact links。
- **8.4 CLI 收敛** ⬜ — `--eval-case` / `--benchmark-suite` / `--retrieval-ablation` / `--geometry-acceptance` / `--material-policy` / `--reference-policy`。
- **8.5 文档收敛** ⬜ — README.md + TODO.md + docs/project_technical_report.md 保持为唯一入口。

---

## 9. 推荐推进顺序

### 当前一条主线：VERA4 从 seed 到从头真实验收

1. **生产路径离线回归**：用 tests 覆盖 MaterialsPatch 写入层 normalization，确认真实 LLM 若再次输出低硼 coolant atom fractions，会被 deterministic repair 修正而不是进入 reviewer/renderer。
2. **复用五 Gate seed 做 no-LLM render/openmc-smoke**：从已通过 Assembled `plan_build_state.json` 进入 render/openmc-smoke，确认 renderer、XML、plot、keff sanity 不依赖重新跑上游 Gate。
3. **运行一次从头真实 LLM VERA4 milestone**：不使用 `--accepted-plan-build-state`，从 Facts→Materials→Universes→Placement→Axial→Assembled→render/openmc-smoke 全链路验证生产流程。
4. **失败处理规则**：若从头 run 未通过，立即停止重复 canary；抽取脱敏 fixture/replay bundle，按 failed stage 分流到 deterministic preflight、production normalizer、prompt/schema 或 reviewer normalization，再离线闭合。
5. **通过后整理 deliverables 与报告**：更新 plots、smoke result、accepted gate chain、material normalization provenance、truth violation summary。

### VERA3 gold model 主线（暂停为次优先级）

VERA3 geometry/material/numerical gold roadmap 保留，但当前优先级低于 VERA4 从头真实生产验收。恢复 VERA3 前，应先确认 VERA4 full production path 不再依赖 seed-only 修复。

### 近期不应并行扩张的方向

在 VERA3 gold model 完成前，不建议同时大规模开展：
- Hex renderer
- depletion
- agent-authored renderer
- long-term memory 自动晋升
- 多模型真实 controlled-route
- 大规模 keff 参数调优

---

## 10. 当前不建议做的事

- 不要把 VERA/C5G7 benchmark facts 写死进 production prompt、validator、renderer 或 guard。
- 不要在 seed/render 路径尚未离线通过前直接反复跑从头真实 LLM canary。
- 不要把 production normalizer 缺口交给 reviewer 消耗真实 LLM 调用；能确定性修复的单位/表达问题必须在 patch 写入或 deterministic preflight 前处理。
- 不要让 RAG/GraphRAG/few-shot 自动确认 density、composition、benchmark constants、loading map 或 nuclear-data path。
- 不要把旧 VERA3 smoke keff 当作 benchmark reference。
- 不要继续使用固定 axial layer count 作为验收指标。
- 不要在 geometry/material acceptance 前通过调密度拟合 keff。
- 不要让 LLM 直接执行未知代码、直接修改 renderer registry 或跳过 deterministic gate。
- 不要把一次成功的真实 LLM 输出直接加入正式 few-shot。
- 不要在 gold model 完成前发布 VERA3 benchmark memory。
- 不要把 spacer grid 整个 open moderator region 替换为结构材料并称为质量守恒。
- 不要在 nozzle / core-plate slab 中同时保留 detailed lattice。
- 不要用 monolithic `reflect_plan` 作为 incremental failure 的默认兜底。
- 不要只验证 assembled plan；必须验证最终 XML 和物理点探针。
- 不要把 XML export 成功等同于 benchmark-accurate。
- 不要创建 solid grid universe 来消除 `grid_cell` 引用错误——spacer grids 必须通过 axial_overlays 表达。

---

## 11. Definition of Done：VERA3 Gold

只有全部满足，才能将 VERA3 标为完成：

### Geometry

- [x] fuel active radial gap 正确
- [x] fuel upper plenum 半径正确
- [x] Pyrex poison / upper gas profile 正确
- [ ] thimble finite profile 正确
- [ ] guide / instrument through-path 正确
- [x] shoulder transitions 正确（lattice + replace_universe_family, 通过 deterministic repair）
- [x] spacer-grid mass/volume 正确
- [ ] nozzle / plate slab 正确
- [ ] axial coverage 无 gap / overlap
- [ ] rendered point probes 全通过
- [ ] geometry debug 无 blocking error
- [ ] annotated plots 人工通过

### Materials

- [ ] UO₂ isotopes 正确
- [ ] coolant variant-specific
- [ ] Pyrex isotopes 正确
- [ ] Zircaloy-4 composition policy 正确
- [ ] SS304 composition policy 正确
- [ ] Inconel-718 composition policy 正确
- [x] nozzle / plate mixture provenance 完整
- [ ] 无 placeholder 被标为 confirmed

### Numerical

- [ ] settings / source convergence 通过
- [ ] keff statistical uncertainty 达标
- [ ] reference keff 对比完成
- [ ] pin power 对比完成
- [ ] axial power 对比完成
- [ ] bias / uncertainty 报告完成

### Agent

- [x] `reference_policy=off` 可生成可审查结构（DeepSeek VERA3A/3B retry_count=0）
- [ ] semantic audit 能发现关键偏差
- [x] unsafe repair 全部拦截（protected-path denylist + clone-only acceptance）
- [x] supervisor 不绕过 blockers（structural blocker precedes human confirmation）
- [ ] gold-vs-agent structured diff 完成

### Knowledge

- [ ] benchmark memory provenance 完整
- [ ] gold plan 版本化
- [ ] anonymous few-shot 人工 review
- [ ] benchmark constants 未泄漏进通用 prompt
- [ ] accepted-run candidate 与 production few-shot 分离
