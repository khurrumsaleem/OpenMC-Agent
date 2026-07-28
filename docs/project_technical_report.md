# OpenMC-Agent 技术报告与进度总览

维护日期：2026-07-28

### 2026-07-28

- **Phase 8C Step 3N VERA4 redraw deliverable refresh**：deterministic VERA4 base fixture now declares explicit radial reflective / axial vacuum six-face core boundaries. Regenerated `deliverables/vera4_plots` from the source-backed fixture path, producing 24 high-resolution PNGs plus an updated contact sheet and manifest. The new render includes RCCA localized-insert slices, center-assembly zooms, spacer-grid planes, fuel-plenum/endplug XZ structure, and upper/lower interface zooms.
- **验证结果**：OpenMC XML export and `openmc -p` plot generation passed in `data/runs/phase8c_step3n_vera4_redraw_v1/workflow`; focused VERA4 fixture tests `59 passed`；全量非 OpenMC/非 LLM pytest `3776 passed, 2 skipped, 393 deselected`，`compileall` 与 fake benchmark `21/21` 通过；baseline diff 因 baseline 文件缺失跳过。

- **Phase 8C Step 3N core smoke / boundary / axial-state closure**：`run_smoke_test` now refreshes stale executable capability from the renderer registry before script generation, allowing accepted `complex_model/core` plans to run instead of failing with `no executable renderer target`. `CoreLayoutPatch` now supports explicit six-face `boundary_conditions`, hierarchical assembly preserves them as `CoreSpec.boundary='mixed'`, and axial preflight blocks finite axial cores that rely on ambiguous all-reflective shorthand. Axial materialization now wires fuel-path-only derived lattices into segment core lattices and honors guide/instrument tube preserve coordinates.
- **验证结果**：focused regression tests `7 passed`；actual VERA4 core smoke through built-in `run_smoke_test` completed locally (`statepoint.5.h5`, `Combined k-effective = 1.02604 +/- 0.03985`). Offline inspection of the prior five-gate seed confirms the high-detail render source still lacks `fuel_plenum/fuel_endplug/water_pin`, uses skeleton/no-material spacer grids, and has `boundary='reflective'` with no six-face boundary conditions.
- **风险/边界**：the previous `keff=1.1835` is not a tuning target; it is produced from an obsolete/incomplete accepted seed with zero axial leakage risk and missing axial structural states. A physically meaningful comparison to reference `keff≈1` requires rerendering from regenerated/qualified axial patches that include source-backed plenum/endplug/grid materials and explicit radial/axial boundary conditions.

- **Phase 8C Step 3M render-detail readiness**：multi-assembly plot derivation now emits high-resolution full-core slices, localized-insert slices, center-assembly zooms, spacer-grid zooms, and upper/lower interface XZ zooms from concrete axial segment metadata. Axial preflight now blocks source-required skeleton/no-material spacer grids and bare component-profile lattice layers without fuel-path loading/profile bindings.
- **验证结果**：focused offline tests `36 passed`；基于 v5 five-gate seed 的离线 reassembly 派生 11 个 diagnostic plot specs 且不调用 LLM；同一 seed 的 axial readiness inspection 报出 skeleton grid/material gap 与裸 `upper_plenum` lattice deterministic blockers，和当前图像细节缺失一致。
- **风险/边界**：本修复不凭空补造 grid/fuel-plenum 材料或尺寸；若上游 Facts/MU 未识别 fuel upper plenum，仍需通过 replay/targeted gate 修正 source-backed 合同后再重新 render。

- **Phase 8C Step 3M render-compile seed handoff closure**：已通过五 Gate 的 Assembled seed v5 可直接进入 render-compile，且不再被 planning `--stop-after-gate assembled_plan` fast-stop 截断。修复 render/openmc stage 的 stop barrier 传递、render-compile final disposition 判定，以及 region expression parser 对 input-derived dotted/percent surface IDs（如 `surf_u_fuel_region_1_2.11pct_fuel_inner`）的支持。
- **验证结果**：focused tests `34 passed`；基于 v5 `plan_build_state.json` 的本地 render-compile seed smoke 通过：`RENDER_COMPILE_CANARY_PASSED`、`renderability=runnable`、`export_backend=real_python_export`、`xml_exported=true`、`llm_call_count=0`。全量非 OpenMC/非 LLM pytest `3757 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21` 通过；baseline diff 因 baseline 缺失跳过。
- **风险/边界**：本修复不重跑真实 LLM gates，不改变物理合同；后续 openmc-smoke 可继续复用同一五 Gate seed，若失败应归类为 OpenMC runtime/geometry smoke 问题而非 Gate 问题。

- **Phase 8C Step 3M openmc-smoke aggregate status closure**：外部 openmc-smoke seed v1 已完成真实 OpenMC 路径（`FIRST_PASS_SUCCESS`、`real_openmc_verified=true`、`xml_exported=true`、`statepoint.5.h5` 存在、`llm_call_count=0`），但 campaign 聚合只识别包含 `PASSED` 的 planning/render milestone，误把 `FIRST_PASS_SUCCESS` 计为 failed run。现 `_campaign_run_passed` 显式接受 `FIRST_PASS_SUCCESS` 与 `RECOVERED_SUCCESS`，同时继续拒绝 failure/incomplete/blocking disposition。
- **验证结果**：focused status tests `48 passed`；外部 v1 `run_result.json` 离线复算 `_campaign_run_passed=True`。全量非 OpenMC/非 LLM pytest `3768 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21` 通过；baseline diff 因 baseline 缺失跳过。
- **风险/边界**：本地 sandbox 复跑 openmc-smoke v2 被 UCX `getifaddrs error: Operation not permitted` 阻断 geometry debug，属于平台权限差异，不覆盖外部 v1 的 OpenMC 成功证据；后续正式 smoke 应在外部命令行新 output dir 运行。

- **Phase 8C Step 3L Assembled reachability preflight closure**：Assembled seed v4 已完成 assembly 并实际调用 reviewer，但 deterministic preflight 将所有未从 selected root 达到的 alternate/segment-specific 对象都写入 `unreachable_required_ids`，产生 457 个 root reachability blocking 假阳性，并诱发 reviewer schema invalid。现 `unreachable_required_ids` 只包含 `ReachabilityRecord.required=True` 且 unreachable 的对象；其他 unreachable 对象进入 `optional_orphan_ids` 审计，不阻断。
- **验证结果**：v4 state 离线重放 Assembled preflight：`ok=True`、blocking `0`、object graph nodes `512`、edges `10348`、renderer `core/runnable`。Assembled focused tests `27 passed`；全量非 OpenMC/非 LLM pytest `3749 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21` 通过；baseline diff 因 baseline 缺失跳过。
- **风险/边界**：本修复关闭 deterministic reachability 假阳性，不忽略真实 required-object reachability；下一次 target run 应直接进入 Assembled reviewer 的真实 finding/accepted 判定。

- **Phase 8C Step 3L Assembled binding planning-scope serialization**：Assembled seed v3 已完成 assembly 并进入 Assembled binding 构造，但在 `AssembledPlanBindingView.planning_scope` 写入 `ResolvedPlanningScope` 对象，Pydantic string 字段校验失败，`llm_call_count=0`。现与 Axial binding 一致，将 resolved scope 提取为稳定字符串值（优先 `value`）。
- **验证结果**：v2 state 本地重放 assembly + Assembled binding 构造成功，scope=`multi_assembly_core`，object graph nodes `240`。Assembled binding focused tests `27 passed`；全量非 OpenMC/非 LLM pytest `3748 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21` 通过；baseline diff 因 baseline 缺失跳过。
- **风险/边界**：本修复只关闭 binding view schema 崩溃，不声明 Assembled reviewer accepted；下一次 target run 应进入 deterministic preflight/reviewer 层。

- **Phase 8C Step 3L Assembled seed skeleton grid materialization closure**：Assembled seed v2 已越过 `axial_overlays.material_missing`，但 assembly 阶段把 skeleton spacer grid 仍送入 concrete grid decoration，触发 `fullcore.grid_density_missing` 与空 `fill_id` schema error。现 materializer/assembler 只对非 skeleton 且具备 material_id 的 spacer grid 执行实体 grid decoration；skeleton overlay 保留在 plan 中供 Assembled Gate 审查，不参与可执行几何物化。
- **验证结果**：v2 `plan_build_state.json` 本地重放 assembly 成功，issue `0`，8 个 skeleton overlay 保留，decorated grid universe `0`。focused tests `125 passed`；全量非 OpenMC/非 LLM pytest `3747 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21` 通过；baseline diff 因 baseline 缺失跳过。
- **风险/边界**：本修复只解决 skeleton overlay 被误物化的问题，不补造缺失 grid material；下一次 Assembled target run 可能进入 reviewer 并将 skeleton grid/source-material gap 作为 blocking finding 分类。

- **Phase 8C Step 3L Assembled seed axial-overlay material normalization**：Assembled seed v1 未进入 Assembled reviewer，失败在补齐 `axial_overlays` 的 patch validation：LLM 生成了 `zircaloy4`/`inconel718`，但 accepted Materials 只含 hashed material IDs，且 Zircaloy-4 有多个候选、Inconel-718 不存在。现新增通用材料 alias 推断（仅唯一匹配才解析）与 axial overlay 生成期 normalization；未知/歧义材料不伪造，降级为 skeleton overlay + human confirmation。
- **验证结果**：新增回归覆盖唯一材料名 alias、歧义 alias fail-closed、生成阶段 unresolved material skeleton 化；focused tests `120 passed`。全量非 OpenMC/非 LLM pytest `3745 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21` 通过；baseline diff 因 `data/evals/workflow/baseline/evaluation_report.json` 不存在跳过。
- **风险/边界**：本修复不补造缺失的 Inconel 材料，也不声明 Assembled Gate accepted；下一次真实验收应使用新 output dir 重跑 `--stop-after-gate assembled_plan`，观察 reviewer 是否将 skeleton overlay 作为 source/material blocker。

- **Phase 8C Step 3L Assembled four-gate seed fixture**：真实 Axial target seed v7 在 commit `164278f` 上通过，`STOP_AFTER_GATE_PASSED:axial_geometry`，Facts/MU/Placement/Axial 四个 gate 均 accepted。新增脱敏最小 PlanBuildState fixture，保留 valid patches 与 accepted hashes，显式设置 Assembled pending。
- **验证结果**：新增回归覆盖 fixture 隐私扫描、`--accepted-plan-build-state` 对 `assembled_plan` 的四 gate accepted 校验、缺 Axial accepted fail-closed、graph 进入 incremental executor、Assembled target seed 上游 owner retry 冻结；focused tests `21 passed`。全量非 OpenMC/非 LLM pytest `3741 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21` 通过；baseline diff 因 `data/evals/workflow/baseline/evaluation_report.json` 不存在跳过。
- **风险/边界**：fixture 不包含 raw prompt/provider/reasoning/API key，不声明 Assembled accepted；下一步真实验收应基于 v7 seed 新目录运行 `--stop-after-gate assembled_plan`。

### 2026-07-27

- **Phase 8C Step 3K Axial reviewer schema-drift normalization**：Axial seed v6 已复用 accepted 上游并通过 deterministic preflight（blocking 0），但真实 Axial reviewer 两次结构化输出均 schema invalid：第一次 severity enum drift，retry 后 findings 使用非规范 code 字段。现扩展 shared review normalizer，支持 `finding_code/issue_code` alias、severity/review_status 大小写与常见 blocking 标签，并修正 `physical_ambiguity` category 拼写白名单。
- **验证结果**：新增 shared review I/O 回归，focused reviewer normalization tests `12 passed`。全量非 OpenMC/非 LLM pytest `3736 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21` 通过；baseline diff 因 `data/evals/workflow/baseline/evaluation_report.json` 不存在跳过。
- **风险/边界**：本修复只提升 structured-review 协议容错；不忽略 finding、不把 Axial reviewer accepted 伪造为通过。下一次仍需新 output dir target-seed Axial canary。

- **Phase 8C Step 3K Axial target-seed deterministic preflight closure**：Axial seed v5 已复用 Facts/MU/Placement accepted 上游并停在 `BLOCKED_BY_GATE:axial_geometry`；失败不是网络或上游 gate 回退，而是 Axial deterministic preflight 将 `assembly_lattice` family alias 判为缺失、将已有 `total_mass_g` 的 spacer grid 误判为缺 density，并忽略已存在的 `localized_insert_profiles`。
- **验证结果**：修复后 v5 `plan_build_state.json` 离线重放 Axial preflight 为 `ok=True`、blocking `0`；focused Axial preflight/retry-policy tests `39 passed`。全量非 OpenMC/非 LLM pytest `3734 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21` 通过；baseline diff 因 `data/evals/workflow/baseline/evaluation_report.json` 不存在跳过。
- **风险/边界**：本修复只闭合 deterministic binding/preflight 假阳性，不声明 Axial reviewer accepted。下一次真实验收仍应使用新 output dir 的 target-seed Axial canary。

- **Phase 8C Step 3K target-seed upstream gate reuse closure**：Axial seed v4 已不再 invalidated Placement-owned patches，但仍在 Axial target run 内重新调用 Placement reviewer，因 reviewer coverage drift 将已 accepted Placement 标成 blocked，继而 Axial barrier 报 `planning.axial_geometry_requires_accepted_placement`。现新增 target-seed upstream gate freeze：accepted seed 下，目标 gate 之前的 accepted stages 无条件复用，不因 hash drift 或循环调用重新 reviewer；Axial target 冻结 Facts/MU/Placement，Assembled target 冻结 Facts/MU/Placement/Axial。
- **验证结果**：新增回归覆盖 Axial target seed 即使 Placement accepted hash stale 也不重审且不触发 placement blocker；相关 focused tests `31 passed`。全量非 OpenMC/非 LLM pytest `3731 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、fixture baseline diff 均通过。
- **风险/边界**：冻结只适用于显式 accepted-state target seed run；普通 canary/resume 仍按 hash drift 重新打开 gate。若 target reviewer 发现 upstream semantic gap，应作为 target blocker/replay 分类，不在该 target run 内重开上游。

- **Phase 8C Step 3K Assembled target-seed routing hardening**：提前审查 Axial seed 系列暴露的 graph/campaign/retry 接线问题在 Assembled Gate 的复现风险。修复 `accepted_plan_build_state + assembled_plan` fast-path：当存在 `stop_after_gate` 时不再直接返回 assembled plan，而是进入 incremental executor，让 Assembled Gate preflight/reviewer 实际执行。Assembled target seed 下所有 patch-owner retry 均被冻结；上游 Facts/MU/Placement/Axial finding 作为 target blocker 输出，不在 Assembled run 内重开上游。run-level `campaign_config.json` 现在记录脱敏 seed 标志与路径，便于排障。
- **验证结果**：新增 Assembled target seed graph routing 与 retry 冻结回归；相关 focused tests `30 passed`。全量非 OpenMC/非 LLM pytest `3730 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、fixture baseline diff 均通过。
- **风险/边界**：本修复只保证 Assembled target canary 的路由和冻结边界正确，不声明 Assembled Gate accepted；真实 Assembled finding 仍需按 replay 分类闭合。

- **Phase 8C Step 3K Axial target-seed upstream freeze closure**：Axial seed v3 已真正使用 Placement accepted seed（`state_id` 与 v16 一致、Facts/MU accepted 复用、LLM calls 降到 3），但 Axial retry 将 upstream Placement-owned patches（`assembly_catalog/core_layout/localized_insert_profiles`）连同 `axial_layers` 一起 invalidated，导致 Placement Gate 从 accepted 退回 pending，并以 `incremental.patch_generation_failed` 收尾。现为 accepted-state target run 写入 seed metadata，并在 retry execution plan 编译阶段冻结 upstream owners：`axial_geometry` target 只允许 `base_path_axial_profiles/axial_layers/axial_overlays` owner 与 Axial Gate replay。
- **验证结果**：新增 target-seed retry 冻结回归，覆盖 upstream owner、canonical task-plan owner fail-closed 与 Axial owner invalidation 限界；相关 focused tests `27 passed`。全量非 OpenMC/非 LLM pytest `3727 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、fixture baseline diff 均通过。
- **风险/边界**：本修复不伪造 Axial 通过；如果下一次 Axial reviewer 仍要求修改 upstream Placement/Facts/MU owner，该 finding 应作为 target run blocker 输出并离线分类，而不是在 Axial run 内重开已 accepted 上游。

- **Phase 8C Step 3K Axial campaign-to-run seed propagation closure**：Axial seed v2 仍未到 Axial；不是网络问题，`failed_network_call_count=0`。顶层 manifest 有 `accepted_plan_build_state_seed=true`，但 per-run `campaign_config.json.metadata={}`，因为 campaign → run config 构造只写入 `llm_budget`，未合并 `campaign.metadata`，导致 graph 初始状态仍没有 seed 并重新生成 Facts/Materials/Universes，最终停在 `BLOCKED_BY_GATE:material_universe`。现已将 campaign metadata 传递到每个 `CanaryRunConfig`。
- **验证结果**：新增 campaign metadata propagation 回归；相关 focused tests `19 passed`。全量非 OpenMC/非 LLM pytest `3724 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、fixture baseline diff 均通过。
- **风险/边界**：本修复只闭合 seed 从 campaign 传到 run 的接线；不声明 Axial accepted。下一次 Axial 验收仍应使用新 output dir（例如 `phase8c_step3k_vera4_axial_gate_seed_v3`），不要 resume v1/v2 失败目录。

- **Phase 8C Step 3K Axial accepted-seed graph handoff closure**：Axial seed v1 失败不是网络或 Axial reviewer 问题；manifest 已记录 `accepted_plan_build_state_seed=true`，但 graph 的 requirement 接收节点未把 `accepted_plan_build_state` 继续传给 generate-plan 节点，导致 seed 分支失效并重新从 Facts/MU 跑起，最终停在 `planning.investigation_materials_blocked`。现已在 `_receive_requirement` 保留 sanitized accepted seed，使 Axial target run 可直接从 Placement accepted `plan_build_state.json` 进入下游。
- **验证结果**：新增 graph seed handoff 回归；Axial seed/readiness/evidence/mode focused tests `17 passed`。全量非 OpenMC/非 LLM pytest `3723 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、fixture baseline diff 均通过。下一次真实 Axial 验收必须用新 output dir（例如 `phase8c_step3k_vera4_axial_gate_seed_v2`），不要 `--resume` 到 v1 失败目录。
- **风险/边界**：本修复只闭合 seed handoff；不声明 Axial Gate accepted，也不改变 resume 的 git/fingerprint 严格校验。若 v2 到达 Axial 后仍失败，应导出脱敏 Axial replay fixture 离线分类闭合。

- **Phase 8C Step 3K Axial Gate offline readiness diagnostic**：新增 `scripts/inspect_axial_gate_readiness.py`，可从 sanitized `plan_build_state.json` 或 `campaign_checkpoint.json` 读取 Axial Gate 前置状态，输出 accepted upstream、required axial patch availability、preflight issue codes、owner routes 与下一步建议，不读取/输出 prompt、raw provider response 或 reasoning。
- **验证结果**：新增 Axial readiness focused tests 与既有 Axial/preflight/stop-after tests 合计 `34 passed`；全量非 OpenMC/非 LLM pytest `3718 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、fixture baseline diff 均通过。对 v16 artifact 离线诊断显示 Facts/MU/Placement accepted，Axial stage 尚未初始化，`base_path_axial_profiles`、`axial_layers`、`axial_overlays` 均无 valid patch，推荐动作是新目录运行 `--stop-after-gate axial_geometry` 生成并验证 Axial patches。
- **风险/边界**：本轮只补诊断和文档，不声明 Axial Gate accepted；不从 v16 resume，因为后续 commit 会改变 `git_sha` 并触发 strict resume mismatch。真实 Axial canary 若失败，应先导出脱敏 fixture 并离线闭合，不连续重跑。

- **Phase 8C Step 3K Axial seed-run preflight hardening**：审查 Axial Gate production path 后修复 applicability 漏洞：Placement accepted checkpoint 中若 Axial component tasks 已排队但 axial patches 尚未生成，`axial_geometry_gate_applicable` 现在返回 true，避免 finalization 误标 `not_applicable`。真实 canary CLI 新增 `--accepted-plan-build-state`，可显式从 sanitized Placement accepted `plan_build_state.json` 启动 Axial/Assembled target-gate run；该入口不放宽 `--resume` 的 git/fingerprint 严格校验。
- **验证结果**：v16 Placement accepted state seed 校验通过，保留 7 个 valid patches 与 Facts/MU/Placement accepted 状态，剥离 Facts/Placement raw review histories；Axial focused/seed tests `29 passed`。全量非 OpenMC/非 LLM pytest `3722 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、fixture baseline diff 均通过。下一次真实 Axial 测试应使用新 output dir 加 `--accepted-plan-build-state data/runs/phase8c_step3j_vera4_placement_gate_v16/runs/run_001/workflow/incremental/plan_build_state.json`，避免重跑 Facts/MU/Placement。
- **风险/边界**：seed 入口要求 requirement 文本匹配、上游 gate accepted；secret-like 字段 fail-closed，raw audit outputs 只在内存中剥离后注入，不提交原始 run 文件。不声明 Axial patch 质量已通过，仍需 target `--stop-after-gate axial_geometry` 验收。

- **Phase 8C Step 3J Placement Gate milestone accepted**：真实 VERA4 `--stop-after-gate placement` v16 在 commit `dc84776b96a93008f74463f650a267a356f396bd` 上完成，enabled gates 为 Facts、Material-Universe、Placement，campaign 结果为 `CAMPAIGN_PASSED`，1/1 successful、0 failed。
- **验证结果**：用户外部运行 artifact 目录为 `data/runs/phase8c_step3j_vera4_placement_gate_v16`，运行时间 `2026-07-27T07:40:47Z` 到 `2026-07-27T09:08:08Z`，模型 `zhipu:glm-5.2`。该 milestone 基于此前同一 HEAD 的全量非 OpenMC/非 LLM pytest `3715 passed, 2 skipped`、`compileall`、fake benchmark `21/21` 和 baseline diff 通过。
- **风险/边界**：Placement Gate 已作为主线里程碑越过；下一主线目标转入下游 Axial Geometry Gate。除 Placement 相关代码再变更外，不再用完整 Placement canary 做日常定位。

### 2026-07-26

- **Phase 8C Step 3J Placement core-layout evidence closure**：Placement v15 已到 Placement reviewer，preflight 无 deterministic issue；reviewer 阻塞 `MISSING_CORE_LAYOUT_EVIDENCE`，原因是 evidence pack 的 `required_patch_types` 包含 `core_layout`，但 `evidence_items` 只包含 assembly scopes、profiles、universes 和 contract rows，没有 core_layout patch fragment。
- **验证结果**：v15 artifact 离线重建 evidence pack 后新增 `C008 core_layout /core_layout`，required patch types 全部有 evidence item；focused Placement/downstream tests `41 passed`。全量非 OpenMC/非 LLM pytest `3715 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、fixture baseline diff 均通过。
- **风险/边界**：本修复只补 evidence coverage，不改变 reviewer、layout semantics 或 renderer。下一次应新目录重跑 `--stop-after-gate placement`。

- **Phase 8C Step 3J Placement shared-profile scope closure**：Placement v14 已到 Facts/MU accepted、profiles 生成成功、reviewer complete 且无 finding，但 deterministic preflight 把 C/E 两个各自合法的 `thimble_plug` intent 互相误判为 `localized_insert.unexpected_assembly_scope`。根因是两个 Facts row 共享同一个 insert kind/profile，而 validator 未检查该 intent 是否已被另一个 assembly-scope row 授权。
- **验证结果**：v14 artifact 离线重放 Placement preflight 从 2 个 `unexpected_assembly_scope` 变为 `ok=True, issue_count=0`；focused Placement/validator tests `58 passed`。全量非 OpenMC/非 LLM pytest `3714 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、fixture baseline diff 均通过。
- **风险/边界**：本修复只闭合 deterministic false positive；不修改 core layout、renderer 或 reactor-specific 规则。下一次应新目录重跑 `--stop-after-gate placement`。

- **Phase 8C Step 3J fragmented Universes localized-insert alias closure**：Placement v13 不是网络失败；Facts/MU accepted，但 run 在 Placement 前的 `localized_insert_profiles` patch generation 阶段失败。根因是 fragmented Universes 只生成 `profile_*` universe 并用 metadata 表示覆盖 localized insert requirement，而 accepted Facts/Profile 合同引用的是 exact `expected_insert_universe_ids`。
- **验证结果**：v13 artifact 离线应用新 alias materialization 后新增 `pyrex_poison/pyrex_plenum/thimble_plug/rcca_aic/rcca_b4c/rcca_plenum/rcca_endplug`，`missing_after=[]`；focused universe/MU/profile tests `58 passed`。全量非 OpenMC/非 LLM pytest `3713 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、fixture baseline diff 均通过。
- **风险/边界**：本修复只保证 Universes patch 暴露下游可引用的 exact IDs；不声明 Placement Gate accepted。下一次应新目录重跑 `--stop-after-gate placement`。

- **Phase 8C Step 3J Placement dependency retry prompt closure**：Placement v12 不是网络失败；Facts/MU accepted，Placement reviewer complete 且无 finding，但 deterministic preflight 缺 localized insert universes，legacy dependency retry 未把 exact required IDs/accepted Facts insert contract 注入 Universes first-attempt prompt，且失败 artifact 吞掉 generation issue 详情。
- **验证结果**：v12 artifact 离线复核中 required IDs 从 `pyrex_plenum/pyrex_poison/thimble_plug/rcca_aic` 补全为包含 `rcca_b4c/rcca_plenum` 的 6 个合同 ID；focused Placement/retry tests `42 passed`。全量非 OpenMC/非 LLM pytest `3712 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、fixture baseline diff 均通过。
- **风险/边界**：本修复使 Universes retry 有完整 source-backed 靶子并保留失败 diagnostics；尚未声明 Placement Gate accepted，需要下一次 `--stop-after-gate placement` milestone 验证。

- **Phase 8C Step 3J Placement review coverage-status closure**：Placement v7 已到 Facts/MU accepted 且 reviewer 返回 `complete_with_gaps`，但 normalizer 只接受 `complete`，错误地把 coverage-complete finding review 记为 `placement_review.coverage_incomplete`。Placement reviewer 现在将 `complete_with_gaps` 作为覆盖完成状态，同时仍要求全部 contract rows 与 evidence refs 显式覆盖。
- **验证结果**：v7 artifact 离线 replay 从 `coverage_incomplete` 变为 `ok=True/coverage_complete=True`，保留 `SEMANTIC_ROLE_NAMING_CONFLICT` 与 `WATER_GUIDE_ROLE_HAS_NO_SUPPLIER` warning findings；focused tests `47 passed`。全量非 OpenMC/非 LLM pytest `3702 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、fixture baseline diff 均通过；默认 `data/evals/workflow/baseline` 路径仍缺失。
- **风险/边界**：本修复只闭合 reviewer coverage-status 协议错误，不声明 Placement Gate accepted；下一次 Placement milestone 应进入 deterministic `localized_insert_profiles` retry/repair，而不是停在 coverage normalization。

- **Phase 8C Step 3J campaign resume failed-run dispatch closure**：Placement v8 resume 没有重跑，原因是 harness 在 `--resume` 时把旧 `campaign_results.json` 中的 failed run 也计为 completed，导致 `pending_runs=[]` 并瞬间复写旧失败。Resume 现在只跳过 `*_PASSED` run；failed/blocked run 保留其 run_dir checkpoint，但会重新调度。
- **验证结果**：v8 result 离线过滤从 `old_results=1` 变为 `kept_results=0/pending_after_filter=[1]`；focused campaign resume/harness tests `32 passed`。全量非 OpenMC/非 LLM pytest `3703 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、fixture baseline diff 均通过。
- **风险/边界**：本修复只保证 resume 会重新 dispatch failed run；如果 provider 继续返回 invalid/ConnectError，仍会停在 Facts investigation，但不再是假 resume。

- **Phase 8C Step 3J Placement multi-assembly owner routing closure**：Placement v9 已到 MU accepted 并进入 Placement review，但 deterministic repair target 同时包含 `pin_map`；multi-assembly/full-core state 没有 valid `pin_map` patch，executor 用裸 `next()` 收集 owner patch 时抛出 `StopIteration`，外层误归 `graph_invocation` infrastructure。Placement deterministic owner routing 现在使用 binding view scope，multi-assembly 不再路由到 `pin_map`；缺 owner patch 也会结构化返回 `planning.placement_repair_owner_missing`。
- **验证结果**：v9 artifact 离线 owner 集合从 `assembly_catalog/localized_insert_profiles/pin_map` 收敛为 `assembly_catalog/localized_insert_profiles`；focused Placement/incremental/downstream tests `48 passed`。全量非 OpenMC/非 LLM pytest `3704 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、fixture baseline diff 均通过。
- **风险/边界**：本修复闭合 harness crash；Placement Gate 仍需继续处理真实 deterministic findings（profile id、anchor、control_state），不声明 accepted。

### 2026-07-25

- **Phase 8C Step 3J Placement retry resolved-state closure**：Placement v6 已进入 Placement review，Facts accepted、Placement review_count=1、网络正常；retry controller 成功提交 Universes owner 并返回 `status=resolved`、`workflow_behavior_changed=True`，但 executor 只对 `resumed` 继续 downstream rebuild，错误地把 `planning.retry.resolved` 当 blocker。executor 现在将已提交 owner 且改变 workflow 的 `resolved/resumed/partially_resolved` 都视为需要 downstream resume。
- **验证结果**：新增 retry outcome 回归覆盖 `resolved + workflow_behavior_changed` 必须 resume、无 workflow change 不 resume。Focused tests `52 passed`；全量非 OpenMC/非 LLM pytest `3700 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、baseline diff 均通过。
- **风险/边界**：本修复闭合 retry 状态机错误，不声明 Placement semantic findings 已清零；下一次 v7 应继续 owner commit 后的 downstream rebuild/Placement rereview。

- **Phase 8C Step 3J Placement profile task-order closure**：Placement v5 使用最新 commit 且网络正常，Facts/MU accepted，但 Placement review 未启动；离线审计显示 `placement_gate_ready=False`，原因是 accepted Facts 声明 `required_profile_id/required_segment_roles`，但初始 task order 未生成 `localized_insert_profiles`。executor 现在从 accepted Facts 推导 profile patch requirement，并在 Facts accepted 后刷新 controlled order/required patch set。
- **验证结果**：新增 accepted-Facts profile contract 与 controlled order refresh 回归，证明 `localized_insert_profiles` 被插入到 `universes` 后、`assembly_catalog/core_layout` 前。Focused tests `38 passed`；全量非 OpenMC/非 LLM pytest `3698 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、baseline diff 均通过。
- **风险/边界**：本修复只保证 Placement gate 的输入 profile patch 会生成；尚未声明 Placement reviewer accepted。下一步用新 output dir 重跑 `--stop-after-gate placement`。

- **Phase 8C Step 3J Placement barrier ordering closure**：Placement v4 真实 run 网络正常、Facts/MU 均 accepted，但在 Placement review 前抢跑 `axial_overlays` 并因 grid material ID/semantic drift 失败。Controlled gate 排序现按 milestone 分层：Facts/Materials/Universes → MU Gate → Placement-owned patches (`assembly_catalog/core_layout/...`) → Placement Gate → Axial patches；`--stop-after-gate placement` 不再依赖 `axial_overlays`。
- **验证结果**：新增排序回归证明 `core_layout` 在 `axial_layers/axial_overlays` 前生成；新增 retry drift 回归允许 schema default 等价变化 `requires_human_confirmation: None → False`。Focused tests `54 passed`；全量非 OpenMC/非 LLM pytest `3696 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、baseline diff 均通过。
- **风险/边界**：本轮未声明 Axial Gate 通过；v4 暴露的 grid material 缺口属于下一阶段 Axial/material contract 问题。下一步重跑 Placement stop-after 应只验证 Placement boundary。

- **Phase 8C Step 3J Placement canary 前置 Facts reviewer scope closure**：`--stop-after-gate placement` v3 未到 Placement，实际在 Facts Gate 被两个 reviewer finding 阻塞：fuel isotope composition 被误归 Facts，空 `operating_state` 被误判为 human-required。Facts staged prompt 与 Python normalization 现将 isotope/composition 归 Materials/MU downstream warning，将显式空 operating state 归 canonical `base` gap warning。
- **验证结果**：新增离线 reviewer normalization 回归固定复现 v3 两类 finding，coverage 不再因 error severity 失败。Focused tests `10 passed`；全量非 OpenMC/非 LLM pytest `3694 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、baseline diff 均通过。
- **风险/边界**：本轮未重跑真实 Placement canary；修复仅闭合上游 Facts reviewer 误阻塞，不声明 Placement/Axial/Assembled accepted。下一步用新 output dir 重跑 `--stop-after-gate placement`。

### 2026-07-24

- **Phase 8C Step 3I Facts blank control-state closure**：v5 MU milestone 未到 MU，Facts Gate 在 revision closure 阻塞 `facts.control_state_contract_missing`；真实 candidate 已包含 `control_state_id=""`，但 deterministic check 用 truthiness 将空字符串误判为缺失。新增 control-state normalization，将空 source state canonicalize 为 `base`，并在 Facts schema、revision merge 与 consistency preflight 复用。
- **验证结果**：v5 最后一轮 `facts_revision_evaluation_002.json` 离线 normalized 后 localized insert control states 全为 `base`，Facts consistency issue codes 为空；focused tests `39 passed`。全量非 OpenMC/非 LLM pytest `3692 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、baseline diff 均通过。

- **Phase 8C Step 3I fuel-variant scoped universe retry prompt**：v3 artifact 离线审计显示 Universes retry owner/target 正确，但 fragment prompt 的 role→material binding 对 fuel role 暴露了所有 fuel materials，LLM retry 后把 Region1/Region2 fuel material 混入同一 fuel universe。修复为按 manifest `fuel_variant_id` 过滤明确不匹配的 fuel material，并在 fragment prompt/schema-repair prompt 与 qualification 中加入 fuel variant material mismatch 约束。
- **验证结果**：v3 目标 `u_fuel_region_2_2.619pct` role binding 离线检查现在只暴露 Region2 fuel material；v3 两个混用 fuel universe 现在在 fragment qualification 级以 `qualification.fuel_variant_material_mismatch` 拒绝。Focused tests `63 passed`；全量非 OpenMC/非 LLM pytest `3689 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、baseline diff 均通过。

- **Phase 8C Step 3I MU final barrier hardening**：v3 canary 中 MU deterministic retry 已生成，但 stage 保持 blocked；executor 在 resume/final path 未强制 Material-Universe accepted，继续进入 assembly 并报 `localized_insert.*` 下游症状。新增 MU accepted barrier，禁止 controlled 模式在 MU 未 accepted 时生成 downstream patch 或 assembly。
- **验证结果**：真实 v3 `plan_build_state.json` 离线 replay 现在秒级停在 `planning.material_universe_gate_not_accepted`（owner `universes`），不再触发 localized insert assembly failure；focused tests `39 passed`。全量非 OpenMC/非 LLM pytest `3686 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、baseline diff 均通过。

### 2026-07-23

- **Phase 8C Step 3I MU deterministic retry routing closure**：v2 canary 已到 Facts accepted，但 MU preflight 阻塞 `u_fuel_region2_2.619/fuel_inner` 误引用 Region1 fuel material；修复 executor 让 deterministic MU blocker 跳过 reviewer 但继续进入 owner retry routing，并让 `universe_id` 写入 retry consumer/required target。修正 failed owner summary 与 completed-failed campaign 仍显示 `CAMPAIGN_RUNNING` 的终态落盘问题。
- **验证结果**：真实 v2 `plan_build_state.json` 离线复现唯一 blocker，并路由到 `owner_patch_types=['universes']`、target `u_fuel_region2_2.619`；focused tests `48 passed`。全量非 OpenMC/非 LLM pytest `3684 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、baseline diff 均通过。

- **Phase 8C Step 3I stop-after run config 回归修复**：修复 `CanaryCampaignConfig.stop_after_gate` 未传入 `CanaryRunConfig` 导致真实 canary 启动时报 `AttributeError: stop_after_gate` 的参数接线缺口；CLI 与 policy 行为不变。
- **验证结果**：stop-after/stage focused tests `22 passed`；全量非 OpenMC/非 LLM pytest `3681 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、baseline diff 均通过。

- **Phase 8C Step 3I MU accepted artifact audit + variant binding preflight**：按 MU accepted 后下游计划提取 bundles 时发现当前 checkpoint 尚无 Placement/Axial/Assembled boundary；继续审计 MU 后 `assembly_catalog` failure，定位为 upstream Universes patch 中 fuel universe `metadata.fuel_variant_id=region1_2.11wt` 但 fuel cells 引用 `source_variant_id=region2_2.619wt` 的材料。修复 MU binding view 读取 universe metadata，并新增 deterministic `material_universe.fuel_variant_material_mismatch`，owner=universes。
- **验证结果**：真实 Step 3H `plan_build_state.json` 离线重跑 MU preflight 现在 fail-closed，blocking 仅 1 个：`fuel_variant_material_mismatch`，指向 `u_fuel_region1_2.11wt/fuel_inner`，expected `region1_2.11wt`、actual `region2_2.619wt`。Focused tests `39 passed`；全量非 OpenMC/非 LLM pytest `3680 passed, 2 skipped, 392 deselected`，`compileall`、fake benchmark `21/21`、baseline diff 均通过；真实 `assembly_catalog_attempt_1_raw` schema+validator 可通过。
- **风险/边界**：本轮撤回“可直接进入 downstream gate live-review”的实现路径；必须先重跑 MU stop-after canary，让 Universes 由 owner retry/regeneration 生成正确 fuel-material binding。未提交原始 run/prompt/raw provider output。

- **Phase 8C Step 3H Shared Insert-Profile MU Preflight Closure**：真实 VERA4 MU checkpoint run 到达 Facts accepted、Materials/Universes valid，但 MU deterministic preflight 阻塞 `material_universe.localized_insert_universe_missing`：同一个 thimble-plug radial profile 服务 C/E 两个 localized insert requirements，旧 `profile_id → insert_requirement_id` 单值映射把 `thimble_plug_C` 覆盖成 `thimble_plug_E`。修复为一对多 localized insert requirement IDs，并让 MU preflight 对旧 checkpoint 通过 GeometryComponentInventory 反查 profile coverage。
- **验证结果**：该真实 run 的旧 `plan_build_state.json` 离线重跑 MU preflight 已 `ok=True`、blocking issues 清零，仅剩 enrichment/background warnings。后续 retry 已到达 MU accepted，但因 `--stop-after-gate` 未真正停止而继续进入 `assembly_catalog` schema/truncation failure；本轮补齐 target gate accepted 后立即 stop、graph save_record 与 `STOP_AFTER_GATE_PASSED:<gate>` 分类。Focused tests `15 passed`（stop-after）+ 既有 Step 3H `32 passed`。
- **风险/边界**：本修复不硬编码 thimble/PWR，仅表达共享 radial profile 可服务多个 localized insert contracts；stop-after 只声明目标 gate milestone，不声明 assembled plan 或 full canary。下一步需重跑同一 MU checkpoint canary，预期 `STOP_AFTER_GATE_PASSED:material_universe`。

- **Phase 8C Step 3G MU Target 验收完成 + 下游离线推进**：composition-basis 修复完成 T0/T1/T2 验收：focused deterministic tests `4 passed`；MU `preflight` replay accepted；v13 `recorded-review` accepted，coverage complete、reviewer_calls=3、schema_retries=0、blocking/rejected finding 均为 0；target-only MU `live-review` accepted，live_review_invoked=True、coverage complete、reviewer_calls=3、issues/findings/rejected 均为空。
- **验证结果**：下游 offline qualification 继续通过，Placement/Axial/Assembled 三个 fixture 的 preflight 与 recorded-review 均 accepted、blocking/rejected finding 为 0；focused downstream tests `30 passed`。本轮未运行完整 Facts→MU canary，避免把里程碑 canary 用作日常定位。
- **风险/边界**：MU target acceptance 证明当前 bundle/reviewer path 已闭合，但不等同于新的真实 Facts→Materials→Universes generation 成功。下一步真实验收应在 production campaign 到达 MU accepted checkpoint 后提取下游 bundles，按 Placement → Axial → Assembled 执行 target-only live-review；三者闭合后再跑一次完整 milestone canary。

### 2026-07-26

- **Phase 8C Step 3J Placement retry targeting**：v11 真实 canary 已越过 MU（Facts/MU accepted）并在 Placement blocked；失败不是网络问题，而是 Placement dependency retry 丢失 missing universe IDs，同时静态 Pyrex/Thimble 插入被 control_state equality 误报。
- **验证结果**：Placement/retry focused tests `21 passed`，downstream replay/retry tests `67 passed`，全量非 OpenMC/非 LLM `3709 passed, 2 skipped, 392 deselected`；`compileall`、fake benchmark `21/21`、baseline regression diff 均通过。
- **风险/边界**：本修复保证 Placement dependency retry 有明确 Universes target，并减少静态 insert false positive；v11 reviewer 仍暴露多段 insert segment universe/material fidelity gap，需下一次 target canary 验证 Universes/Profile regeneration 是否闭合。

### 2026-07-22

- **Phase 8C Step 3G 下游 Gate 恢复循环 + Live-Review 编排**：(1) 新增 `tests/test_downstream_gate_recovery.py`（10 tests）覆盖 Placement retry_controller 完整循环（resolved/no-progress/budget/stale）和 Axial/Assembled re-replay 恢复（block→mutate→accepted、rejected fail-closed、deterministic replay）。(2) 新增 `scripts/extract_downstream_replay_bundles.py` 从真实 `campaign_checkpoint.json` 提取 `production_accepted` 下游 bundle。(3) 新增 `scripts/run_downstream_live_review.py` 按 Placement→Axial→Assembled 顺序运行 preflight+review（支持 live/recorded/preflight 三模式、early-break/continue-on-fail），MU accepted 后可直接使用。
- **验证结果**：Step 3G focused tests `18 passed`（recovery 10 + live-review 8）；`compileall -q openmc_agent scripts` 通过。全量验证与 baseline diff 见下文。
- **风险/边界**：恢复循环测试用 fake producer/validator，不声明真实 provider 验收。extract 脚本依赖真实 campaign 到达下游 boundary snapshot，当前所有真实 canary 均未到达 placement。live-review 编排脚本的管道已用 offline bundle 验证（recorded-review 模式），真实 live-review 需等 MU accepted。
- **验证结果**：全量非 OpenMC/非 LLM 测试 `3707 passed, 2 skipped, 392 deselected`；`compileall`、fake benchmark `21/21`、baseline regression diff 均通过。fraction basis 现在只允许 normalized fraction (`0 < sum <= 1`)；percent-style vectors (`≈100`) 在 fragment qualification 中确定性 `/100` 归一化，未经过 qualification 的 bad patch 由 MaterialsPatch/MU preflight fail-closed；`materials.composition_fraction_sum_invalid` 注册为 materials-owned deterministic gap。
- **风险/边界**：不改变 renderer 或 OpenMC transport semantics；LLM 仍可选择 `stoichiometric_ratio` 表达化学式。修复后需重跑 target Placement gate canary，预期该问题不再消耗 MU reviewer finding 循环。

- **Phase 8C Step 3D 下游 Gate Replay**：accepted checkpoint 边界扩展至 Placement/Axial/Assembled；GateReplayBundle 现在支持五个 gate，下游 bundle 强制保存脱敏 policy snapshot 与完整 accepted upstream chain。preflight、recorded-review、live-review 复用目标 gate 的生产 preflight/evidence/reviewer，并输出 coverage、blocking/rejected finding、terminal status 与脱敏 diagnostics；新增三条 gate accepted 原子 checkpoint 回调。
- **验证结果**：checkpoint/replay/downstream focused tests `45 passed`，`compileall -q openmc_agent scripts` 通过；未重跑完整 VERA4 canary，Step 3C canary 已有终态且未到达下游 gate，故本阶段仅完成代码级 replay 验收。
- **风险/边界**：旧 Facts/MU bundle 保持可读；下游 gate 缺少 policy snapshot、上游 accepted 状态、canonical hash 或包含敏感字段时 fail-closed。live-review 仍仅调用目标 reviewer，默认上限 1800 秒；尚未声明 Placement/Axial/Assembled 真实 provider 验收。

- **Phase 8C Step 3E 下游 Gate 离线预资格**：新增三个 `offline_deterministic` 脱敏 replay fixture、fixture fingerprint、离线资格 CLI 与 Placement/Axial/Assembled mutation corpus。资格入口按目标 gate 顺序复用生产 preflight 与 recorded-review，报告 coverage、blocking/rejected findings、terminal status；checkpoint/resume 覆盖三条新增 accepted boundary。
- **验证结果**：Step 3E focused offline qualification tests `53 passed`；三个 clean fixture 的 preflight/recorded-review 均 accepted、coverage complete、blocking/rejected 均为 0。未运行真实 LLM canary，未将合成 upstream chain 表述为 MU 或 VERA4 acceptance。
- **风险/边界**：mutation 证明 deterministic blocker/owner route 稳定性，不替代真实 provider 验收。MU accepted 后仍需依次运行 Placement → Axial Geometry → Assembled target-only live-review，每次最长 1800 秒；三个 live-review 闭合后才运行一次完整 milestone canary。

- **Phase 8C Step 3F 五 Gate 离线 Campaign 恢复资格**：新增脱敏版本化 `CampaignRecoveryScenario`、clean 五 gate recorded campaign 与恢复故障矩阵。runner 复用 checkpoint/store、gate replay、生产 downstream resume seam 和 dependency graph，覆盖漂移、损坏、敏感字段、缺失上游、timeout/schema/finding blocker 及上游 patch 变更闭包。
- **验证结果**：Step 3F focused recovery tests `13 passed`；离线 matrix `15` 场景，clean accepted、故障场景受控 blocked，输出仅含 hash、边界复用/失效、调用计数和稳定 issue code。未运行真实 LLM/OpenMC，不将结果表述为 MU 或下游 provider acceptance。
- **风险/边界**：该阶段证明恢复与失效传播，不证明真实 provider 或 renderer 资格；Step 3D Facts canary 仍独立运行。后续真实验收仍需 MU accepted 后按 Placement → Axial → Assembled target-only live-review 顺序执行。

- **Phase 8C Step 3D Facts Revision Stale-Finding Closure**：T3 full VERA4 canary 已实际完成 1 run，但未到 MU：`BLOCKED_BY_GATE:facts`，`planning.facts_revision.unresolved_requires_human`，Facts blocked、MU pending、truth violations 0、15 real LLM calls。根因是 Facts revision rereview 后的 closure metadata 仍引用初审旧 count findings，且 reviewer 将“coverage confirmation, not an error”自相矛盾输出保留为 non-repairable error。
- **验证结果**：修复 Facts reviewer confirmation-not-error 降级为 warning，并让 Facts revision blocked metadata/registry 使用最新 rereview unresolved findings；新增脱敏 Step 3D fixture，不含 prompt/raw provider output/reasoning/secrets/原始 canary 日志。Focused Facts/GateReplay tests `58 passed`；全量非 OpenMC/非 LLM pytest `3620 passed, 2 skipped, 392 deselected`；`compileall -q openmc_agent scripts`、fake benchmark `21/21` 与 fixture baseline diff 均通过。
- **风险/边界**：本修复只处理 Facts revision closure 状态机与 reviewer normalization，不改 MU、Materials/Universes generation、renderer 或物理合同。修复后需先跑 Facts target-only live-review；Facts accepted 后再重跑 `--stop-after-gate material_universe` T3。

- **Phase 8C Step 3C MU Replay-driven Finding Closure**：新增脱敏 v13 MU recorded-review fixture（3 个 split scopes、6 个 normalized finding target，不含 prompt/raw provider output/reasoning/secrets/原始 canary 日志），GateReplay recorded-review 现在按 materials/universes/binding scope 独立回放。MU normalization 拒绝重复 deterministic preflight issue、已闭合 stale finding、role-only contract 过度具体化、scope/contract row 错配；新增 classification/diagnostics（reviewer_false_positive、deterministic_preflight_gap、materials/universes_contract_gap、binding_metadata_gap、true_source_gap，未知 code fail-closed）。修复 fuel variant coverage 按 `source_variant_id` 选材，U235 composition fraction 统一为 wt% 后比较。
- **验证结果**：focused MU/replay tests `62 passed`；v13 fixture `preflight` clean，`recorded-review` coverage complete、blocking finding count 0，原 6 个 recorded findings 中 2 个 deterministic repeat、1 个 stale matrix finding、1 个 role-only over-specific finding 被 normalization 关闭，剩余 2 个 warning 分别分类为 reviewer_false_positive / binding_metadata_gap。全量非 OpenMC/非 LLM pytest `3617 passed, 2 skipped, 392 deselected`；`compileall -q openmc_agent scripts`、fake benchmark `21/21` 与 fixture baseline diff 均通过。MU target `live-review` 因平台外部网络策略改由用户本地执行，JSON 终态为 `live_review_invoked=True`、`review_ok=True`、`coverage=True`、`reviewer_calls=3`、`schema_retries=0`、`findings=[]`、`rejected=[]`、`blocking_finding_count=0`，materials/universes/binding 三个 scope 均 complete。
- **风险/边界**：本阶段不改变 renderer 能力边界，不硬编码 VERA4/PWR 规则；T3 触发后发现新的 Facts revision closure blocker，因此 Step 3C 不声明 full canary accepted，MU 仍以 recorded/live closure 为当前稳定证据。

- **Phase 8C Step 3B 修正状态**：accepted boundaries 按 `gate:facts` -> `patch:materials` -> `patch:universes` -> `gate:material_universe` 排序。checkpoint/store/callback 全部 fail-closed；hydrate 区分无 snapshot 与漂移/损坏，验证 `state_hash` 和 requirement/input/policy/git/structured-output hashes。Facts action checkpoint 只保存 hash、规范化 planner/tool progress、最小 source spans 与 claim；恢复会先校验上下文指纹，再重建 ledger/index，跳过已完成 action，provider timeout/未完成 action 单独重跑。
- **验证结果**：全量非 OpenMC/非 LLM pytest `3611 passed, 2 skipped, 392 deselected`，`compileall -q openmc_agent scripts`、fake benchmark `21/21` 与 fixture baseline regression diff 均通过。Facts 与 MU 各完成一次 target-only `live-review`（最长 1800 s）；Facts 仅返回 source-gap warning，MU 完成 3 个 review scope 并产生可复现 finding 终态。workflow regression baseline 已补为 `tests/fixtures/workflow_baseline/evaluation_report.json`，默认 fake diff gate 不再依赖被忽略的 `data/evals/workflow/baseline/`。
- **风险/边界**：MU live finding 来自最小 v13 fixture 的未证实 composition/implicit-role 数据，尚未完成 replay 分类与闭合；因此按分层策略没有运行完整 VERA4 canary。replay 只支持 Facts 与 Material-Universe 两 gate，不改变材料、universe、物理或 renderer 合同。

- **Phase 8C Step 3A MU Gate 可达性与恢复协议**：新增 gate checkpoint 数据合同与原子 JSON store，accepted gate 复用同时校验 input/evidence/inventory/structured-output policy/canonical hashes；任一漂移 fail-closed。Facts investigation provider deadline 独立记录为 `provider.timeout`，保留已计费调用、deadline、payload hash，禁止隐藏 schema retry；未完成 session 不写入可复用 cache。新增 checkpoint/timeout 回归覆盖；MU deterministic preflight 与 v12/v13 binding 回归保持零错误。
- **验证结果**：focused MU/investigation/replay tests `24 passed`；全量非 OpenMC/非 LLM `3579 passed, 2 skipped, 392 deselected`；compileall、fake benchmark `21/21` 与 `git diff --check` clean。真实 VERA4 canary 已启动但在 provider 调用前被 `BLOCKED_BY_LLM_ENVIRONMENT` 阻塞（`glm` API 环境未配置），未宣称 gate 完成。
- **风险/边界**：本次只加强恢复与真实性 telemetry，不改变材料、universe、几何或 renderer 合同；provider 持续超时仍报告为外部运行阻塞，不可宣称 MU Gate 完成。

- **Step 3A 重配 canary 分析**：扩大 Facts/MU closed-loop budget 到 review=4、repair=4、additional calls=60，并保留 Facts→Materials→Universes→MU reviewer 路径。当前后台 retry 尚未形成终态 artifact；中间 Facts 证据显示 scope/fuel/assembly 已通过，但 completeness 仍有 boundary/end-plug provenance gaps。已确认 checkpoint 仍需从最终 artifact writer 前移到 action/gate 完成时。

维护方式：每完成一个重要工程 Step 后更新本报告的"当前状态""验证结果""风险/边界""下一步建议"和"维护记录"。**维护记录使用精炼风格**：每条 2–4 行（日期 + 主题 + 核心改动 + 测试数），不写冗长根因/实现细节（那些在代码与 git history 里）。

### 2026-07-22

- **Phase 8C Step 3 MU Canonical Binding — 确定性漂移修复 + Preflight Blocking**：分析 v9 MU Gate 16 个 preflight issues 的 4 个根因，实施 7 项修复。(1) UniverseSpecPatch 新增 `metadata` 字段 + 碎片化 pipeline 在 merge 后注入 `geometry_profile_id`/`source_requirement_ids`/`localized_insert_requirement_id`/`fuel_variant_id`；(2) `validate_materials_against_requirement_set` 修复 dict-vs-model 访问 bug；(3) MU preflight `localized_insert_universe_missing` 增加 profile-based fallback；(4) `extract_universe_requirements_from_inventory` 交叉引用 `localized_insert_profiles` 设置 `localized_insert_requirement_id`；(5) MU gate controlled 模式下确定性 preflight error 直接 BLOCK（不调用 reviewer）；(6) 扩展 inventory profile 合同：fuel pin 声明 cladding/coolant/gas 支持 roles，localized inserts 声明 cladding/coolant；(7) 修复 dedup：localized insert bindings 和 material role reqs 在 profile 合并后重映射到 surviving profile_id。Canary v12 验证：Materials **18/18** ✓、Universes **5/5** ✓、MU preflight 仅 **3 errors**（从 v11 的 32 降至此），全部为确定性修复目标（fuel_variant_material_duplicate + localized_insert_universe_missing）。v13 修复后 3 errors 目标已消除（等待 canary 验证因 LLM 超时而未完成）。3575 passed / 2 skipped；benchmark 21/21。

- **Phase 8C Step 2F-1 Inventory-First Universe Contract Closure — MU Gate REACHED**：在真实 GLM-5.2 VERA4 canary v9 中首次到达 Material-Universe Gate。核心改动：controlled 模式下 universes 只消费 `InventoryUniverseRequirementSet`（不再 emit `implicit:*`），增加 material-role preflight（`unavailable_material_role` blocker 在 LLM 调用前阻塞不可满足合同），fragment prompt 含 role→material_id binding，第二次尝试用 schema-repair prompt。Canary v9 (`c876abd`)：Facts accepted、Materials **6/6**、Universes **5/5**（全部 first-attempt，0 failure）、34 LLM calls、~58min。MU Gate review_failed（15 findings，2 repair attempts，预期迭代闭合）。同时修复 latent bug：`_maybe_execute_research_for_gate` 缺 `PlanStageStatus` import（v7 触发）。18 new tests；3565 passed / 2 skipped；compileall clean；benchmark 21/21。

- **Phase 8C Step 2F Facts→MU Canary（4 轮迭代修复）**：在真实 GLM-5.2 VERA4 canary 中验证 Facts→Materials→Universes 全链路。经过 6 次 canary 迭代发现并修复 4 个 P0/P1 问题：(1) materials fragment pipeline 使用 bare `json.loads` 无法处理 markdown fence / CoT — 加 `parse_llm_patch_json` fallback；(2) fragment prompt 未列有效 enum 值，LLM 返回 `mass_frac` 而非 `weight_frac` — 加显式 enum 约束；(3) Facts rereviewer 将 `/missing_facts` finding 标为 `repairable_by_llm=False` — 对 soft metadata fields 覆盖该分类；(4) LLM 对 structural material 返回 `composition_status='library'` 等 synonym — 加 `_normalize_material_enum_synonyms` 预处理。最终结果（canary v6, `0d4c1ed`）：Facts accepted、Materials **6/6** accepted、Universes **8/9** accepted（失败: `implicit_gas_gap` parse+qualification）、MU gate 未到达、0 truth violations、23 LLM calls、60.3min。Universe `implicit_gas_gap` fragment 是 LLM 难以生成的 implicit geometry element。3547 passed / 2 skipped；compileall clean；benchmark 21/21。

- **Phase 8C Step 2E Facts Gate Closure — Real VERA4 Canary ACCEPTED**：`5889817` 三轮 closure transaction（round-0 修 4 个 aggregate count fields: pyrex=80/thimble=112/guide=216/instrument=9；round-1 修 spacer_grid=72；rereview 全 stage coverage 通过）在真实 GLM-5.2 VERA4 canary 中首次通过 Facts Gate。21 LLM calls、35.7min、0 truth violations、0 human intervention。此前 phase8b_step4b2_canary_run1/run2 均 `BLOCKED_BY_GATE:facts`。Canary 使用 `--stop-after-gate facts`，MU/Placement/Axial/Assembled gates 未跑（pending）。`data/runs/phase8c_step2e_canary_v2/`。

- **Phase 8C Step 2E Facts Gate Closure Implementation**：Facts revision 改为 bounded 三轮 closure transaction，每轮使用 shared two-attempt structured-output path + canonical input payload hash + clone validation + duplicate/no-progress protection + full rereview before durable commit。只有 rereview complete coverage 且无 error finding 才 commit；partial 但 valid candidate 作为下一轮的 immutable input。Facts completeness reviewer 将 downstream Materials/Placement/Axial/Universe 需求标记为 nonblocking warning（`downstream_impact`），不再阻塞 Facts Gate。Investigation 在 baseline coverage 完成时跳过 planner（`skipped_after_coverage_complete`）。57 focused regression + full non-OpenMC/non-LLM suite + compileall + benchmark 21/21。

- **Phase 8B Step 4B-2 Materials Fragment Transaction Stabilization**：新增 `materials_fragment_generation.py`（deterministic manifest 从 `MaterialGenerationRequirementSet` 构建，per-item `contract_hash`；fragment qualification 验证 identity/role/variant/placeholder/poison-absorber/mixture-components；`AcceptedMaterialFragmentRecord` checkpoint with hash/contract integrity；`merge_material_fragments_structured` 纯 Python merge with fragment/manifest/global scope 归因）和 `materials_patch_pipeline.py`（`generate_materials_patch` 编排器，auto monolithic→fragmented 切换 on truncation；checkpoint/resume 只重生成损坏 fragment；targeted fragment replay for merge failures；provider call classification 无静默 `except: pass`）。executor dispatcher 在 `planning_material_requirement_set` 可用时路由 materials 到 fragmented pipeline。历史截断根因：monolithic materials patch 在 attempt 1 schema_error（assumptions 类型错误）、attempt 2 json_parse_error（JSON 未闭合），retry 重新生成全部 materials。31 新测试。3535 passed / 2 skipped；compileall clean；benchmark 21/21。真实 GLM-5.2 LLM 2-material focused test 成功（fuel+coolant fragment qualification 全 passed，session completed=True）。

- **Phase 8B Step 4B-1+ Facts Gate Path Normalization & Revision Trigger**：Canary run2 暴露两个 P0 bug：(1) staged prompt 展示 `facts_subset` dict → LLM 输出 `affected_json_paths=["facts_subset.expected_pyrex_count"]`（无 `/` 前缀）→ normalizer 全部拒绝（`path_out_of_scope`）→ 14 个 finding 丢失（含 LLM 正确提供的 expected_value=80/112/72）；(2) executor 在 `coverage_complete=False` 时立即早退返回 → `compute_allowed_actions`（本可返回 `REVISE_CURRENT_PATCH`）永远不执行 → `repair_count=0`。修复：(1) `_normalize()` 在 scope 检查前自动归一化 `facts_subset.X→/X`、裸`X→/X`（幂等，`/materials`、`/universes` 仍被拒绝）；prompt 增加 JSON Pointer 格式指引。(2) executor 只在 `review.ok=False` 时早退；`coverage_complete=False` 但 review 成功 → 继续到 action-decision → revision 触发。20 新测试。3504 passed / 2 skipped；compileall clean；benchmark 21/21。

- **Phase 8B Step 4B-1+ Facts Gate Coverage Aggregation Fix**：修复 Canary run_004 暴露的 Facts Gate 阻塞——LLM 返回 `review_status=insufficient_evidence`（语义≈"完成但有 warning"）被旧 `last_status == "complete"` 逻辑错误阻塞。P0：`_REVIEW_STATUS_FIXES` 新增 `insufficient_evidence→complete_with_gaps`、`completed/reviewed→complete`；`_ACCEPTED_REVIEW_STATUSES = {complete, complete_with_gaps}`；`_aggregate_coverage` 替换 last-stage 检查，要求所有 stage 在 accepted set 中且无 error-severity finding（fail closed for unknown）。P1：`raw_sink` callback 在 provider 返回边界捕获 raw text（`StructuredOutputAttempt` 仍不保留 raw content），`StructuredReviewResult.raw_outputs` + `FactsReviewResult.raw_outputs` 现包含真实 LLM 响应——修复 `facts_review_raw_*.json` 永远为空的 telemetry bug。5 个 review_status Literal 新增 `complete_with_gaps`。28 新测试。3492 passed / 2 skipped；compileall clean；benchmark 21/21。

- **Phase 8B Step 4B-1 Universe Fragment Transaction Qualification**：新增 `universe_fragment_qualification.py`（schema/identity/kind/cell-role/material-ref/material-role/placeholder/duplicate-cell-id/canonical-hash 确定性 qualification），`AcceptedFragmentRecord` 持久化 data+hash+contract_hash+qualification_status；resume 逐项重新验证、只降级损坏 fragment。`UniverseManifestItem` 新增 localized_insert_requirement_id/base_path_component_profile_id/protected_through_path_roles 并计算 per-item `contract_hash`；`merge_universe_fragments_structured` 返回 `UniverseMergeResult`（fragment/manifest/global scope 归因 + `invalid_fragment_ids`），保留 `patch_generation.merge_failed` 顶层码以兼容现有 retry owner policy。`generate_universes_patch` 在 transaction 内实现 targeted replay（fragment-scoped failure 只重放 invalid fragments，manifest/global fail closed），LLM 调用分类为 provider_exception/parse_exception/empty 而非静默吞掉。确定性复现 run_004 类失败（`implicit_gas_gap` 中 `material_id="REPLACE"` 占位符）。57 新测试；3464 passed / 2 skipped；compileall clean；benchmark 21/21。详细文档 `docs/phase8b_step4b1_universe_fragment_transaction.md`。

### 2026-07-20

- **Phase 8B Step 4 Material-Universe Gate Stabilization**：新增 `MaterialRequirementSkeleton`/`UniverseRequirementSkeleton` typed models（source-derived identity/role/variant/profile/protected-path，5 统一 resolution states，编译自 Inventory+Facts+Evidence，不交 LLM 决定存在性/数量/角色/variant）；Facts accepted 后自动编译 skeleton 并注入 Materials/Universes generation constraints。扩展 deterministic preflight：density missing、compound isotope unresolved、localized insert universe missing（携带 required_ids+affected_json_paths 给 targeted retry）、protected path missing（仅 Facts+Inventory 声明 protected_through_path_roles 时检查）、fuel variant collapse per-universe、requirement skeleton coverage。拆分 MU reviewer 为 Material/Universe/Binding 3 个独立 scope call（独立 schema/fingerprint/artifact，coverage 要求 3/3 complete）。修复 `SourceExcerpt` evidence_hash latent bug（MU/Axial/Assembled reviewer 直接注入 EvidenceItem.canonical_hash 违反 hash 语义）。注册 `patch_generation.merge_failed` 到 universes owner。17 新测试。3406 passed / 2 skipped；compileall clean；benchmark 21/21。真实 canary：run_002 preflight 正确发现 7 个 missing insert universes，retry owner=universes 正确；run_003 split reviewer universes/binding scope 通过、materials scope schema 失败（已修复 scope payload）；run_004 universes merge failure（pre-existing patch generation 问题）。连续 3 次 MU Gate accepted 尚未达成。

- **Phase 8B Step 3 Facts Gate Stabilization**：新增 `facts_evidence_consistency.py` 确定性 evidence↔FactsPatch 一致性检查（scope/fuel_variant/localized_insert/spacer_grid 冲突不交 LLM）；拆分 Facts Reviewer 为 6 typed stages（SCOPE/FUEL_VARIANT/ASSEMBLY_STRUCTURE/LOCALIZED_INSERT/GRID_AXIAL/COMPLETENESS），每 stage 只看相关 FactsPatch 子集，prompt 从 ~38KB 降至 ~8–15KB；repair prompt 增加 REQUIRED COVERAGE FIELDS 强制覆盖，repair candidate 缺字段直接 `facts_revision.incomplete_coverage` block；reviewer 空响应→`facts.reviewer_empty_response`、自由文本 approve→`facts.reviewer_free_text_approve`。21 新测试。3393 passed / 2 skipped；benchmark 21/21。
- **Phase 8C Step 2D real-canary closure**：修复 investigator recorder bypass（`_PromptOnlyWrapper` 未拦截 `generate_patch_json`→虚假 `real_llm_not_verified`）；将 tool/argument 验证整合进 structured-output transaction 的 `_normalize`，消除 `argument_invalid` 立即阻塞；`planning_network_call_count` 计入 `plan_investigator` role；新增 Facts 语义合成步骤（LLM 从工具证据合成 `model_scope`/`fuel_variant` 等谓词 claims，闭合 coverage gap）。GLM-5.2 真实 canary 验证：truth_violations 清空、Facts investigation 完成（6 claims 合成、coverage_complete=True）、Facts patch 生成有效、Facts Gate 可 ACCEPTED。3372 passed / 2 skipped；benchmark 21/21。
- **Phase 8B Step 3 real-canary follow-up**：在 `--stop-after-gate facts` 的 GLM-5.2 真实运行中，6 个 staged reviewer 输出均成功解析且 `review_status=complete`，Facts Gate deterministic decision 为 `approve`、stage status 为 `accepted`；campaign 顶层 `PLANNING_FAILURE` 仅表示未继续执行后续 planning gates，不代表 Facts Gate 失败。该 run 使用旧 SHA `1817e1e`，因此只作为行为证据，不作为当前 HEAD 的最终资格声明。全量测试 3394 passed / 2 skipped；compileall clean；fake benchmark 21/21。
- **Facts revision RFC6902 append fix**：当前 HEAD 真实 canary 进一步暴露 `add /assumptions/4` 被 JSON Patch evaluator 错误拒绝的问题；数组 `add` 的 numeric index 等于当前长度时应允许 append。修复 `_array_index` 并新增回归测试；真实 proposal 已从 `apply_failed: 4` 变为可成功应用。验证：3395 passed / 2 skipped；compileall clean；fake benchmark 21/21。
- **Current-HEAD Facts Gate stability evidence**：在 `51cee9f` 上连续 3 次 `--stop-after-gate facts` + GLM-5.2 真实运行（run_004/005/006）均达到 `facts=accepted`，6 个 staged reviewer 均返回 `review_status=complete`，未再出现 `facts.reviewer_empty_response` 或 `facts_revision.apply_failed`。三次顶层 `PLANNING_FAILURE` 都发生在 Facts accepted 之后的后续 patch-generation，不计入 Facts Gate failure；因此 Facts Gate stabilization 可声明通过，但 full planning canary 仍未通过。raw review artifacts 仍为空而 normalized/manifest 有内容，保留为审计可观测性缺口。

### 2026-07-19

- **P2-PLAN-INVESTIGATION Phase 8B Step 1**：修复 Inventory 与 Material-Universe retry owner registry 双根因：14/14 inventory codes 从 retry_owner_policy() 返回 None→全部解析（materials/universes/special routes）；11/37 MU codes 补齐（借 canonical material_universe_issue_owner 适配层）。新增 SpecialRetryRoute 类型（RETRIEVE_EVIDENCE/INVENTORY_REBUILD/ASK_HUMAN），消除 plan_investigation/inventory_rebuild 伪 Patch。新增 MaterialBindingSkeleton + UniverseBindingSkeleton（编译时 python 注入 requirement_id/role/variant_id/profile_id）。16 个新 truthfulness codes。3240 passed / 2 skipped；benchmark 21/21 100%。

### 2026-07-19

- **P2-PLAN-INVESTIGATION Phase 8A Step 7**：完成 research evidence synthesis 全闭环：candidate_spans_found vs evidence_added 状态语义分离（Section 4）；LLM synthesis 严格 JSON contract（proposals/unresolved/conflicts，LLM 只能引用系统提供的 span_id）；10 条 proposal 验证规则；commit_research_evidence_proposals 生成 EvidenceClaim + 改变 ledger_hash；ResearchCompilationDiff + ResearchInvalidationPlan（仅失效受影响 patch，Facts/Placement/Axial 永不失效）；MU Gate wiring：RETRIEVE_EVIDENCE → deterministic search → LLM synthesis → ledger commit → gate reopen，或 no_evidence_found → scoped patch revision。VERA4 MU canary v5 验证 research pipeline 正确触发（research_requested → no_evidence_found，4 个 blocking findings 是 Patch content 问题非 source 缺失）。新增 3 测试文件 / 25 tests，16 个新 truthfulness codes。3188 passed / 2 skipped；fake benchmark 21/21。
- **P2-PLAN-INVESTIGATION Phase 8A Step 6A**：修复 7 个 P0 controlled-fail-closed 缺口（P0-1 Materials/Universes investigation 缺失 → 通用 `run_patch_investigation_stage(patch_type=...)`；P0-2 inventory compile 异常非阻塞 → 受控阻断；P0-3 preflight except→[] → typed `InventoryPreflightExecutionResult` + 显式 owner map；P0-4 EvidenceConstraintPayload 替代伪 EvidenceClaim；P0-5 `InvestigationContext` 加 typed inventory 字段；P0-6 `enabled_gates_through` 累计前缀；P0-7 recursive resume 传递全部 kwargs）。新增 5 测试文件 / 39 tests。VERA4 MU canary 验证：Materials + Universes investigation 实际运行 + evidence 注入 + inventory 编译（6 mreq / 5 ureq）+ MU preflight 正确 surface 4 blocking findings。
- **P2-PLAN-INVESTIGATION Phase 8A Step 6B foundation**：合约 0.8→0.9 + `RETRIEVE_EVIDENCE` 枚举；新增 5 个 typed research models（Target/Request/Result/Delta/AbsenceRecord）；确定性 router（LLM 不参与路由）+ 最小化 executor（复用 investigation tools，no shell/network/grep）；MU Gate wiring 通过 `_maybe_execute_research_for_gate` 在 blocking 前执行 RETRIEVE_EVIDENCE。新增 4 测试文件 / 47 tests。
- **P2-PLAN-INVESTIGATION Phase 8A Step 6C**：PlacementRequirementSet + AxialGeometryRequirementSet + 2 evidence-qualified preflights（13+20 stable codes）+ GateReviewAssignment（assignment-based coverage，reviewer self-report 不再是唯一依据）+ 30 个新 truthfulness codes（Section 32）。新增 3 测试文件 / 38 tests。Shared SourceIndex + Ledger 每个 incremental run 只构建一次（Section 7）。3163 passed / 2 skipped；fake benchmark 21/21。

### 2026-07-17

- **P2 Plan Closed-Loop Facts Gate Phase 1B**：Facts Gate contract 升级至 0.3；新增 `review_failed`、完整 EvidencePack+schema retry、JSON-object recovery、structured-mode telemetry、review/validation/budget ledger；Facts revision 改为 clone→re-review→atomic commit，human ambiguity 改为 namespaced LangGraph interrupt/resume 并以 typed confirmed fact 驱动 facts regeneration。历史 VERA4 critic 失败已复盘为非 JSON 输出与无上下文 retry，尚未重跑真实资格 canary。验证：Facts/closed-loop/graph targeted tests。
- **2026-07-17 · P2 Placement Gate Phase 2**：closed-loop contract 升级至 0.4；提取通用 structured review I/O，并新增单/多组件 binding view、contract matrix、静态预检、Placement Critic、issue ownership、受限多 patch clone revision 与通用 plan-gate human routing。验证：placement/facts targeted tests；真实 LLM/代理资格未声明，dependency retry 与 Final Plan Gate 仍未实现。

- **P2 Plan Closed-Loop Phase 0**：新增 reactor-neutral typed protocol、SHA-256 semantic fingerprints、deterministic gate/action/state-machine policy，以及持久化到 `PlanBuildState` 的独立预算和 no-progress ledger。advisory 仅写 JSON artifacts；off 不进入框架；controlled 明确 fail-closed 为未实现。
  验证：新增 closed-loop targeted tests（11 passed）；未新增 Critic/Repair/Supervisor LLM 调用或 human interrupt。边界：Facts Gate 及后续 executable repair loop 尚未实现。

## 1. 项目定位

OpenMC-Agent 的目标是把自然语言反应堆建模需求转成可审查、可校验、尽可能可运行的 OpenMC Python 模型。系统的核心安全边界是：

- LLM 只生成结构化 `SimulationPlan`，不直接写最终运行代码。
- 本地 Pydantic schema、validator、renderer 和 OpenMC 工具链负责校验、渲染、导出和 smoke test。
- 缺失材料密度、composition、核数据库路径、benchmark 常数、真实装载图等事实缺口必须保留 human confirmation，不能由 RAG/GraphRAG 自动补全。

当前仓库已经从"结构化建模 agent"扩展成"带诊断闭环、检索编排、GraphRAG、知识注入、incremental 分层 plan 生成、few-shot 双轨、trace/evaluation 和 benchmark 基础设施"的中大型工程。

## 2. 代码规模

截至本报告维护时：

- `openmc_agent/` Python 文件：62 个，约 35,200 行。
- `tests/` Python 测试文件：53 个，约 23,200 行。
- 文档：本报告 + 10 个活跃策略文档。
- 最近全量测试：`3240 passed, 2 skipped, 392 deselected in 86.60s`；compileall clean；benchmark 21/21 100%。

## 2a. 重大里程碑（2026-07-09）

**VERA3 3A 和 3B 通过 incremental plan builder 端到端成功运行**（真实 LLM deepseek:deepseek-chat）：

- 两个变体全部 7 个 patch 由 LLM 直接生成（不依赖 reference fixture）。
- pin_map 24 个特殊坐标 → assembler 确定性展开 17×17=289 full lattice。
- CellLayerPatch 几何 → assembler 自动构建 ZCylinder surfaces + regions（同心圆柱 pin cell）。
- 边界条件从实际几何推导（radial=reflective, axial=vacuum）。
- 元素符号（He/Zr/Fe/Ni）自动路由到 add_element。
- coord_overlap 确定性修复（3B 同坐标保留高优先级组）。
- OpenMC smoke test 通过。

| 指标 | 3A | 3B |
|---|---|---|
| keff | ~1.149（偏高，合金近似） | **0.979 ± 0.004** |
| Leakage | 0% | 低 |
| Surfaces | 6 | 10 |
| Regions | 9 | 15 |
| Pin counts | 264F + 24GT + 1IT | base: 264F + 24GT + 1IT; finite Pyrex/plug loading requires component-profile support |

## 3. 架构总览

核心流程：

```text
User requirement
  -> _receive_requirement            # 检测 incremental 触发条件，写 planning_mode_decision
  -> retrieve_openmc_docs            # 本地内省 OpenMC API
  -> select_few_shots                # 抽象提纲 + gold case（结构特征打分）
  -> generate_plan                   # dispatcher:
       monolithic  : LLM 一次输出 SimulationPlan（简单 case）
       incremental : plan_builder 7-patch 分层生成 + 确定性组装（复杂 3D case，默认）
  -> validate_plan / capability
  -> retrieval orchestrator          # grep / graph / GraphRAG planner / GraphRAG / RAG / ranker
  -> reflect_plan / ask_expert / auto_repair
  -> renderer                        # PinCell / RectAssembly / Core / TRISO / Skeleton
  -> export_xml / plot / smoke test
  -> trace / records / evaluation
```

主要模块职责：

| 模块 | 职责 |
| --- | --- |
| `schemas.py` | Pydantic IR：`SimulationPlan`、materials、geometry、lattice、axial overlay、capability、issues。 |
| `graph.py` | LangGraph workflow：plan 生成（monolithic/incremental dispatcher）、validate、capability、repair、ask_expert、render、tools、trace。 |
| `plan_builder/` | incremental 7-patch 分层 plan 生成：mode 判断、PlanBuildState、patch schemas、validators、deterministic assembler、LLM patch generator、per-patch prompts、executor + retry router、LLM adapter、reference-backed patches、evaluation harness。 |
| `few_shots.py` / `few_shot_cases.py` | few-shot 选取（抽象提纲 + gold case，结构特征打分）+ gold case loader（slim IR / patch exemplar / 结构特征，堆型无关）。 |
| `assembly3d_guard.py` | 3D assembly 需求被压扁为 2D 的 requirement/plan 级 guard。 |
| `axial_overlay.py` | Level 1 `homogenized_open_region` overlay 渲染决策（堆型无关，保 pin/tube through-path）。 |
| `source_settings.py` / `geometry_bounds.py` | source/bounds 一致性校验（source z 绑定活性燃料、xy 绑定 full footprint、plot 覆盖）。 |
| `verification.py` | verification digest（不变量检查表 + pin/axial/overlay/material/bounds）+ 3D voxel plot。 |
| `validator.py` / `lattice_validation.py` | 结构校验、pin map / lattice 诊断、错误码生成。 |
| `error_catalog.py` | 稳定 issue taxonomy、route hints、repair/retrieval/human confirmation hints。 |
| `auto_repair.py` | deterministic patch，优先于 LLM reflect。 |
| `grep_search.py` | 受控 grep 检索。 |
| `knowledge_graph.py` / `knowledge_graph_registry.py` | hand-written graph registry + `graph_lookup`。 |
| `rag_search.py` | 本地 lexical RAG（无外部服务）。 |
| `graphrag_retriever.py` / `graphrag_query_planner.py` | Graph-guided RAG + issue-intent query planner。 |
| `knowledge_ingestion.py` / `knowledge_runtime.py` | 本地 docs/examples ingestion + knowledge graph 运行时加载。 |
| `retrieval_orchestrator.py` / `evidence_ranker.py` | 统一检索编排 + evidence scoring/dedup/budget。 |
| `workflow_trace.py` / `evaluation.py` / `benchmark_runner.py` | trace model/recorder/export + evaluation case/result/metrics + benchmark/ablation runner。 |
| `renderers/` | PinCell、RectAssembly、Core、TRISO、Skeleton 渲染器和 registry。 |
| `tools.py` | OpenMC export_xml、plot、smoke test 子进程工具。 |

## 4. 已完成能力

### 4.1 诊断闭环

- runtime / export_xml / hex lattice 错误码化；`ValidationIssue` 结构化；`ToolResult.issues` 接入 workflow。
- `reflect_plan` / `ask_expert` 按 issue route 处理；hex lattice 保持 skeleton，不伪装 runnable。
- workflow 有稳定 issue code、schema path、concept id、route hint、retrieval/human confirmation 标记。

### 4.2 检索层

- grep（精确定位代码/测试/示例/文档）、graph（schema/concept/API/doc/error/repair 关系）、RAG（本地 chunk + lexical scoring，无外部服务）。
- GraphRAG Retriever（issues → graph expansion → RAG request）+ Query Planner（issue intent → start nodes/depth/filters/queries）+ Evidence Ranker（merge/dedup/score/budget）。
- 默认链：`issues → grep → graph → GraphRAG planner → GraphRAG → RAG → merge → ranker → reflect_plan prompt`。manual review / fact gap 也触发文档检索，但 human confirmation 保留。

### 4.3 Knowledge Ingestion + Runtime Loader

- `Input/knowledge_sources.json` manifest；扫描 `docs/`、`examples/`、`openmc_docs/`、`openmc_examples/`、`Input/`；规则式 annotation；chunk → graph nodes/edges；JSON/JSONL 输出。
- **Runtime Loader**（2026-07-07）：orchestrator 在 GraphRAG stage 通过 `RetrievalPolicy.knowledge_graph_path` 或 `OPENMC_AGENT_KNOWLEDGE_DIR` 加载持久化 graph 作 `extra_nodes/extra_edges`；失败只 warning 不中断。详见 `docs/knowledge_runtime_strategy.md`。

### 4.4 Trace / Evaluation / Benchmark

- `TraceRecorder`/`WorkflowTrace`/`TraceEvent`；validation/retrieval/auto-repair/reflect/ask_expert/render 摘要事件。
- `EvaluationCase`/`EvaluationResult`/`EvaluationMetrics`；benchmark/ablation runner（fake runner、JSON/JSONL/MD 输出）。

### 4.5 Renderer 与能力边界

当前 renderer：PinCell、RectAssembly、Core、TRISO、Skeleton fallback。

明确未完成或受限：

- HexAssemblyRenderer 未实现。
- depletion / burnup、pebble_bed renderer 未实现。
- Level 1 spacer-grid overlay 是均质近似（详见 4.8），**非** volume-fraction calibrated；不建模真实格条/混流翼/套筒。
- fact gap 仍必须走 ask_expert / human confirmation。

### 4.6 Incremental Plan Builder（Phase 0–7D）

复杂 3D assembly（多 variant / spacer grid / special pin map / 大 lattice / JSON parse 失败历史）从 monolithic 25K JSON 改为 **7-patch 分层生成**（`facts → materials → universes → pin_map → axial_layers → axial_overlays → settings`）+ 确定性组装：

- 每层独立生成/校验/重试；`PinMapPatch` 只输出特殊坐标（不展开 289 格）；`settings` 确定性。
- graph dispatcher：`should_use_incremental_planning` 触发时路由到 incremental executor，简单 2D 仍走 monolithic；`use_incremental_executor=True` 默认。
- **Input-driven structural patches by default**：复杂模型默认按未见模型处理，由 incremental patch LLM 分层生成 structural patches；reference JSON 仅作为显式策略使用（policy：`off` 默认 / `prefer_reference_for_structural` / `reference_only_for_structural` / `fallback_after_llm_failure`）。
- **Resumable**：`save/load_plan_build_state`，已 valid patch 跳过。
- **Plan-level targeted repair**：assembled `SimulationPlan` 通过 schema 但被 validator 拦截时，graph 将 issue code / schema_path 映射回 patch root，executor 只失效该 patch 及下游依赖后定点重做；无法定位时才退回 fresh regeneration。
- **真实 LLM 加固**：禁止不安全 monolithic fallback、strict patch-output contract（禁 full plan / full lattice 输出）、structured output mode（`json_schema`）、per-patch attempt artifact 可见。

### 4.7 Few-shot 双轨增强（2026-07-08）

- 4 个结构命名的 gold case（`data/few_shot_cases/`：`pin_cell_basic` / `assembly_2d_lattice` / `assembly_3d_with_spacer_grids` / `quarter_core_with_reflector`），全部 **anonymize 掉堆型标识**（VERA/C5G7/CASL → `[reference]`/`EXAMPLE`），数值为 illustrative。
- **Incremental 路径**：`build_patch_prompt` 注入 patch 形态参考段（受 2400 字符/层预算）。
- **Monolithic 路径**：`_augmented_plan_requirement` 注入 slim IR + digest。
- 选择按**结构特征**（kind/lattice_size/axial_overlay/reflector/quarter）打分，绝不按堆型名——通用性自检测试守护。
- 边界：仅 3D assembly case 有 patch few-shot（有现成 fixture）；其余 case 只有 monolithic slim_ir。

### 4.8 3D assembly guard + Level 1 overlay（Step 1–6）

- `assembly3d_guard.py`：requirement 级 + plan 级检测 3D 需求被压扁为 2D（四个 `assembly3d.*` issue code）。
- `AxialOverlaySpec` IR + Level 1 `homogenized_open_region` overlay renderer（保留 pin/tube through-path；导向管多 open cell 保守复用）。
- skeleton overlay 自动提升（数据齐全时从 skeleton 升 Level 1）。
- source/bounds 一致性：source z 绑定活性燃料、xy 绑定 full footprint、plot origin 重定心（修 quarter-plot bug）。
- verification digest + 3D voxel plot（堆型无关结构化验证产物）。


### P0 Evaluation Backbone: case schema, workflow runner, and trace metrics

P0 的目标是建立可量化的真实 workflow 回归基准，使每次 RAG / GraphRAG / incremental / renderer 改动都能定位指标变化和失败 stage。当前实现范围是 plan-only lightweight workflow case runner：默认不跑 OpenMC export、plot、smoke test 或 transport run，只收集 planning / validation / capability / artifact 诊断 trace。

新增指标覆盖 `plan_schema_success`、`incremental_patch_success`、issue precision/recall、retrieval trigger、artifact completeness 与 planning mode accuracy。后续 P0-D/E 再正式接入 benchmark runner 运行入口、CLI、markdown/json report、ablation 与 real LLM opt-in benchmark。

### P0-NEW-1：受控材料 composition policy

**目的**：把 Zircaloy-4 / SS-304 / Inconel-718 从纯元素近似（pure Zr / Fe / Ni）升级到 nominal 合金成分（含 Sn/Cr/Ni/Nb/Mo/...），为 keff 对比建立可量化 baseline。

**为什么纯元素近似不安全**：把合金简化成单一基体元素会丢失真实的吸收体（Sn、Cr、Ni、Nb、Mo），导致 keff 系统性偏高。这会让"能跑"和"跑得准"无法区分，使 keff 对比失去诊断价值。

**实现**：

- 新增 `openmc_agent/material_library.py`：`AlloyComposition` registry + alias resolver。覆盖 `zircaloy4` / `ss304` / `inconel718` 三个 canonical id，每个成分 sum=1.0，`source_note` 明确声明是 nominal engineering approximation（不是 VERA 官方 spec），可整体替换。
- 新增 `openmc_agent/material_policy.py`：`MaterialCompositionPolicy` 枚举（`preserve_plan` / `apply_alloy_library` / `strict_confirmed_only`），默认 `apply_alloy_library`。Policy 仅在 (a) material id/name canonicalize 到已知合金，且 (b) 当前 composition 是该合金的纯元素近似（单元素或空）时才替换；fuel / water / helium / pyrex / unknown alloy 全部保留原样，不 block。
- assembler 接入 policy：`_assemble_materials` 现在返回 `(materials, issues, MaterialCompositionReport)`，每次替换写一条 `materials.alloy_library_applied` info issue；assembled plan 同时产出 `material_composition_report.json`（写入 `incremental/`），记录每个材料的 alloy_id / elements / policy。
- graph + executor 透传 `material_policy` 参数（默认 apply_alloy_library）；workflow case runner 的 `_artifact_keys` 把 `material_composition_report` 视为合法 artifact key。
- 新增 `scripts/compare_material_policies.py`：dry-run / OpenMC 两种模式，输出 `comparison_report.json`（含 `preserve_plan` keff、`apply_alloy_library` keff、`delta_pcm`）。

**安全边界**：composition 来自公开 handbook midpoint，**不是** benchmark-specific 常数；density 仍由 plan/patch 提供；fuel/water/pyrex 不替换。smoke run 不是 benchmark agreement。

## 5. 当前 Retrieval/GraphRAG 状态

### Query Planner

`plan_graphrag_query(...)` 把 issues 分类为 `schema_repair` / `runtime_diagnosis` / `export_xml_repair` / `lattice_map_repair` / `renderer_capability` / `documentation_lookup` / `fact_gap_review` / `benchmark_interpretation` / `unknown`，按优先级输出 `GraphRagQueryIntent` / `GraphExpansionPolicy` / start nodes / preferred queries / required filters / avoided queries / planned paths。

### Evidence Ranker

`rank_and_select_evidence(...)`：dedup same locator/doc_chunk_id/near-duplicate；grep exact > graph relationship > GraphRAG > plain RAG；按 issue/schema/concept/API/graph path/ingested node 加分；fact gap unsafe evidence 降分；控制每类数量和 prompt 总字符数。

### Prompt 输出

默认有 ranking 结果时：`[GraphRAG Query Plan] [Graph Context] [Ranked Evidence] [Evidence Safety Constraints]`。


### Environment validation and OpenMC test gating

Base Python 环境允许不安装 OpenMC；测试分层为 `test-no-openmc`（pure Python / no runtime）、`test-openmc`（需要 OpenMC Python package/runtime）和 `test-all`（完整环境）。OpenMC 相关测试统一用 `openmc` marker 与 `pytest.importorskip("openmc")` gate，缺失 OpenMC 时应 skip 而非 collection error。

新增 `scripts/check_environment.py` 作为环境自检入口：默认只报告 Project/OpenMC/OPENMC_CROSS_SECTIONS/Conda/Micromamba 状态且 OpenMC 缺失仍 exit 0；`--require-openmc` 才在缺失 OpenMC 时返回非 0。

## 6. 验证状态

```bash
conda run -n openmc-env python -m pytest -q
# 841 passed, 3 skipped in 46.74s
```

覆盖：schemas / llm / graph / renderers / executor / validator / assembly3d_guard / axial_overlay / source_settings / geometry_bounds / verification / plan_builder（mode / state / patches / validators / assembler / patch_generator / patch_prompts / executor / retry_router / llm_adapter / reference_patches / evaluation）/ few_shots / few_shot_cases。3 skip 为真实 LLM / integration gated（CI 不跑）。

## 7. 当前已知边界和风险

### 7.1 工作区风险

工作区存在未纳入工程提交的用户脏文件：`Input/VERA1_problem.md`–`Input/VERA5_problem.md`、`Input/CASL-U-2012-0131-004.pdf`。这些是用户维护的输入资料，提交工程代码时应排除。

### 7.2 技术风险

- Incremental 已是复杂 3D 默认路径；VERA3 3B 和未来复杂模型默认都按未见模型走 input-driven structural patch synthesis，reference-only 仅用于显式 gold/reference 回归。真实 LLM 生成 `pin_map` / `axial_layers` / `axial_overlays` 的稳定性仍需持续评估（真实 LLM 测试 opt-in，CI 不跑）。
- Level 1 overlay 是均质近似，非 volume-fraction calibrated；不建模真实格条/混流翼。
- **VERA3 component-profile gap**：canonical contract 已确认燃料端塞/气腔是 fuel-pin internal geometry，Pyrex/套管塞是有限轴向 guide-tube inserts。P0-D5 已统一 validator/renderer structural issue source、提前在 validate_plan 阶段拦截 component-profile material slab、新增 shoulder_gap role、实现确定性 multi-patch repair bundle；真实 VERA3 3A smoke 不再因 shoulder gap 进入 reflect/retry。
- 材料/合金 composition 仍多为 approximate（Zircaloy-4 / SS-304 / Inconel-718 简化为纯元素 + warning）。
- patch few-shot 仅 3D assembly case 有；其余 case 的 incremental 路径用泛型 `_PATCH_RULES`。
- Query planner / evidence ranker 是 heuristic，未真实 benchmark 权重校准。
- Benchmark runner 还没推进到真实 workflow case runner；无 persistent trace store / dashboard。

### 7.3 安全边界

RAG / GraphRAG / ingested docs / ranked evidence / few-shot 都只能作为上下文：不能自动确认 nuclear data path、材料密度/composition、benchmark constants、真实 loading map。few-shot 数值为 illustrative reference，不是事实确认来源。

## 8. 下一步建议（更新于 2026-07-09）

VERA3 3A/3B 端到端成功后，重心从"能不能跑通"转向"跑得准不准"：

1. **P0-NEW keff 精度提升**（最高优先级）：
   - 3B keff=0.979（接近临界，非常好），3A keff≈1.149（偏高 ~15%）。
   - 主要偏差来源：合金近似（纯 Zr/Fe/Ni 去除了吸收截面，导致 keff 偏高）。
   - 下一步：引入受控材料 composition library（Zircaloy-4 Sn/Fe/Cr, SS-304 Cr/Ni, Inconel-718 Cr/Fe/Nb/Mo），替代纯元素近似。这不违反安全边界（composition 来自公开核数据手册，不是 benchmark-specific 常数）。
   - 同时考虑：更高粒子数（当前 5 batches），轴向反射层是否需要额外处理。

2. **边界条件验证**（已加入 TODO P1-5）：
   - FactsPatch 增加显式边界字段，validator 增加边界合理性检查。
   - 渲染后自检 XML boundary_type 与 plan 对照。

3. **真实 evaluation case runner**：从 fake trace 推进到 lightweight workflow，评估 retrieval trigger rate / fact gap preservation / skeleton-runnable 分类 / issue code precision/recall。

4. **Volume-fraction calibrated overlay**：grid 体积/质量标定，提升 spacer grid 物理保真度。

5. **Full VERA3 keff benchmark acceptance**：用更高粒子数跑完整 criticality 计算，与 benchmark 参考值对标。
5. **patch few-shot 补齐**：从 IR 反推 VERA2A / C5G7 / pin_cell 的 patch few-shot。
6. **真实 LLM incremental 稳定性**：多模型 ablation。
7. **VERA3 component-profile production IR**：在不枚举 264 个 fuel 坐标的前提下，扩展 component axial profile 与 per-layer composable lattice loading；renderer 必须从 base guide-tube through-path 派生有限 Pyrex/thimble insert，且不让 nozzle homogenization 与详细 lattice 重叠。

## 9. 维护记录

> 精炼风格：每条 2–4 行（日期 + 主题 + 核心改动 + 测试数）。详细根因/实现见代码与 git history。

### 2026-07-16 (MULTI-ASSEMBLY-DETECTION)

- **多组件堆芯特征检测缺失修复**：VERA4（3×3 九组件）跑完 7 个 patch 后在 assemble 阶段报 `assembly.missing_patch`（缺 `assembly_catalog`/`core_layout`）。根因：`should_use_incremental_planning`（`plan_builder/mode.py`）只有 axial/spacer/special-pin/variants/large-lattice 五类特征，**没有 multi-assembly 检测**，导致 `feature_summary.multi_assembly_core` 永远为空 → task order 走单组件路径（含 pin_map、无 assembly_catalog/core_layout），而 facts patch 声明 `model_scope=multi_assembly_core` → assembler 要这俩 patch → 失败。新增 `_detect_multi_assembly_core`（堆型无关关键词 + "N assemblies"/"九个…组件" 计数正则）+ `TRIGGER_FEATURE_MULTI_ASSEMBLY`，VERA4 现在走多组件路径 `[facts, materials, universes, assembly_catalog, axial_layers, axial_overlays, core_layout, settings]`（去掉顶层 pin_map）。单组件/pincell/VVER 单组件负例不误触。非-openmc `1938 passed`，benchmark `21/21`。

### 2026-07-16 (PATCH-NULL-COLLECTION-COERCION)

- **patch schema 容忍 LLM 写的 null 集合字段**：官方 `deepseek-chat` 越过 facts/materials/universes/pin_map 后卡在 `axial_layers` 的 `schema_error`——每层 `loading_ids: null`，但 pydantic `list[str] = Field(default_factory=list)` 只在字段**缺失**时用默认、显式 `null` 会被拒。在 `_PatchBase` 加 `model_validator(mode="before")`，对所有 list/dict 类型字段把显式 `null` 归一为 `[]`/`{}`（通用、堆型无关，避免逐字段打地鼠；与 `schemas.py:708` 运行期模型既有做法一致）。attempt-1 的 `role` 枚举不匹配是单次问题、retry 已修正，不放宽枚举。非-openmc `1935 passed`，benchmark `21/21`。

### 2026-07-16 (JSON-EXTRACT-COT-RECOVERY)

- **从推理模型 CoT 中恢复 JSON patch**：`ds:deepseek-v4-flash` 无视 `reasoning_effort=low`，仍先吐 ~16K 字符 CoT 再给 JSON，旧贪婪正则 `\{[\s\S]*\}` 因 CoT 里含散落 `{` 而抓不到合法 JSON（facts 阶段 `json_parse_error`）。新增 `_scan_json_objects`（在每个 `{` 处 `json.JSONDecoder().raw_decode`）+ `_pick_best_json_object`（取最后一个 `patch_type` 匹配的完整对象），从 CoT 后正确回收 patch；无 `patch_type` 匹配则回退旧贪婪路径、不误收 CoT 示例。同时把诊断 raw 切片 5000→50000（`executor` patch_attempt_artifacts），便于看清推理模型完整输出。非-openmc `1932 passed`，benchmark `21/21`。

### 2026-07-16 (DS-REASONING-TOKENBUDGET)

- **SenseNova `ds:` 推理模型输出截断修复**：`ds:deepseek-v4-flash` 默认开思考模式，CoT 经 `reasoning_content` 吃光输出 token 预算导致 JSON 写一半即断（facts 阶段 `json_parse_error`/`full_plan_output_forbidden`）。最终落地两处：(1) `DSChatClient.adjust_payload` 注入 `reasoning_effort`（默认 `low`，`SENSENOVA_REASONING_EFFORT` 可覆盖）；(2) 新增 `_looks_truncated` 括号平衡启发，截断时报独立码 `patch_generation.json_truncated`（带"降 reasoning_effort/显式调大 max_tokens"提示）而非笼统 `json_parse_error`。
- **回退 max_tokens 自动 cap（回归修复）**：曾把 `PATCH_MAX_TOKENS` 接进 `executor→generate_patch`，但官方 `deepseek-chat` 的 universes patch 实测 ~15237 字符（≈6000 token），4500 的 cap 直接把它截断 → 回归。改为**不主动设 max_tokens**，用各 provider 默认（DeepSeek ~8192，比任何安全统一 cap 都大）；`PATCH_MAX_TOKENS` 保留为参考预算表（universes 标注 6500），可显式传给 `generate_patch`。`generate_patch` 仍保留 `max_tokens` 形参供按需使用。非-openmc `1930 passed`，benchmark `21/21`。

### 2026-07-16 (PINMAP-COUNT-SCOPE-FIX)

- **pin_map 计数 scope 误报修复（multi-assembly core_total vs per-assembly）**：多组件堆芯（VERA4，9 组件）下 pin_map 校验器拿 facts 的 legacy `expected_*_count`（core_total：2376/216/9/80/112）直接比对单组件 pin_map 实际值（264/24/1/20/28），全部误报 `patch.pin_map.count_mismatch`。新增 `scoped_counts.resolve_expected_counts_for_pin_map`：从 `assembly_type` scoped 条目按角色归一到 per-assembly 叠合 scope（跨型号相同取公共值、型号专属取和），`executor` facts 分支在多组件时用它覆盖 `expected_counts`。非均匀 core 且无 assembly_type 细分的 role 不强制（避免误报）。非-openmc `1928 passed`，benchmark `21/21`；真实 VERA4 pin_map 端到端校验 0 issues。

### 2026-07-16 (PINMAP-FULL-LATTICE-FIX)

- **pin_map 全格阵误报修复（结构化判定）**：`patch_generator._detect_full_lattice` 由原始文本启发式（`len(raw)>3000` 或括号总数 `>80`）改为解析 JSON 后按*单个*坐标列表最大长度相对 `lattice_size` 格数判定（≥50% 才算近满格 dump），新增 `_pin_map_largest_coord_list` 助手。修复 VERA4 多插入件 pin_map（24 guide tube + pyrex/thimble/RCCA 合计 ~97 坐标、raw>3000 字符）被误判为全格阵并导致 plan 阶段中止的 false positive；禁用全格阵字段名（`universe_pattern` 等）仍由 `validate_patch_contract` 兜底，raw 50K backstop 仅作 DoS 防护。非-openmc `1924 passed`，benchmark `21/21`；真实 VERA4 raw 复跑 `generate_patch` ok=True。

### 2026-07-15

### 2026-07-16 (LOGGING-MIGRATION)

- **CLI Logging Migration: Python logging + default-visible progress**：迁移运行时进度消息（`[node:...]`/`[llm] ...`/`[normalize] ...`）从手动 `print(file=sys.stderr)` + 布尔开关到标准 `logging` 模块；新增 `logging_setup.py`（`configure_logging(level)` + `OPENMC_AGENT_LOG_LEVEL` env var），`graph.py` `_progress()` 改用 `_logger.info()`（95 个调用点签名不变），`llm.py` `_llm_log`/`_normalize_log`/`_StderrStatus._enabled`/heartbeat 改用 `_logger.isEnabledFor(INFO)` 门控，`set_llm_progress()` 保留为 backward-compat wrapper；CLI 入口添加 `--log-level`/`--quiet`（`run_model.py` 默认 INFO、benchmark 默认 WARNING），Makefile 添加 `LOG_LEVEL` 变量；conftest.py autouse fixture 设 WARNING 静默测试，`test_inspect.py` `[node:]` 断言迁移到 `caplog`。非-openmc `1923 passed`，OpenMC `388 passed`，benchmark `21/21`。

### 2026-07-16 (GRID-ACCEPTANCE-CLOSURE)

- **P2-FULLCORE-2D-A-GRID-ACCEPTANCE-CLOSURE: Strict Grid Geometry Gate, False-Positive Rejection, Material Probes**：新增通用 `grid_geometry_validation.py`（10-point validator: decorated universe 非空 → lattice 引用 → IR 合并 → frame cell → grid material → region → surface → material 可达性 → grid-on/off digest 差异 → dangling refs），fail-closed 传播到 `PlanAssemblyResult.ok`；`build_grid_geometry_reachability_report()` 层次遍历（axial layer → core lattice → assembly → pin lattice → decorated universe → frame cell → region → material）；VERA4 acceptance 扩展 Level F（12 grid checks: 8 bands, 72 instances, 2 end/6 middle, decorated universes, lattice refs, frame cells/regions, material catalog, assembly gap, validator, reachability）；XML integrity gate 检查 model.py + materials.xml + geometry.xml 中 grid objects 完整性；structured material probes（frame/inner-mod/pin/gap/grid-off position probes）；reactor-neutral synthetic fixture（2×2 core, 3×3 pin lattice, 3 grid bands）；5 negative false-positive fixtures（A-E）；7 new test files（78 tests）。非-openmc `1923 passed`，OpenMC `388 passed`，benchmark `21/21`，VERA4 acceptance `46/46`（含 12 grid checks），grid-on/off digest 不同（9a86... vs 1736...），smoke keff=**1.27150±0.00346** zero lost。**P2_FULLCORE_GRID_GEOMETRY_INJECTED**。**P2_FULLCORE_GRID_REACHABILITY_VALIDATED**。**P2_FULLCORE_GRID_FALSE_POSITIVE_GATE_READY**。**VERA4_DETERMINISTIC_GRID_CLOSURE_PASSED**。

### 2026-07-16 (GRID-CLOSURE)

- **P2-FULLCORE-2D-A-GRID-CLOSURE: Physical Spacer-Grid Universe Injection**：关闭 grid geometry 最后一公里：materializer 主循环在 base path state → localized inserts 之后调用 `_make_grid_decorated_universe()`，对 grid-active segment 中每个 universe ID 生成 grid-decorated variant（square_frame + bounded background），替换 derived lattice pattern 中的 universe ID 引用，grid band z-boundaries 加入 global breakpoints 使 segment 在 grid 边界正确分割；assembler `_assemble_universes()` 处理 grid-decorated UniverseSpecPatch，background region 排除 frame area（`+cyl ~ ( frame_expr )`），surface ID 使用 `lo/hi` 替代 `+/-` 避免 region expression parser 冲突，region expression parser 修复 `~` 前缺少 `&` 问题。VERA4 结果：22 grid-decorated universes（覆盖 fuel/​guide_tube/​inst_tube/​pyrex/​thimble/​rcca 全部 pin-state），4 unique grid geometries（Inconel end + Zircaloy middle），176 frame surfaces，86 universes / 207 cells / 67 lattices / 275 surfaces / 156 regions，model.py 806KB → 5 XML → geometry loads → transport smoke keff=**1.2715±0.00346** → zero lost。cross-process identity **10/10**。非-openmc `1845 passed`，OpenMC `388 passed`，benchmark `21/21`。**P2_FULLCORE_GRID_GEOMETRY_INJECTED**。**VERA4_DETERMINISTIC_GRID_CLOSURE_PASSED**。

### 2026-07-16 (HARDENING)

- **P2-FULLCORE-2D-A-HARDENING: Physical Fuel-Path States, Grid Geometry, Host-Path Validation, Deterministic Identity**：关闭四个已知缺口：(1) Base fuel-path axial-state materialization——新增 `BasePathAxialProfilesPatch`/`BasePathStateBindingPatchItem`，materializer 根据 `segment.base_role` 切换燃料路径 universe（lower_shoulder→water_pin, lower_endplug→fuel_endplug, upper_plenum→fuel_plenum），R1/R2 燃料差异在 active_fuel 保持，guide/instrument tube 在所有状态中连续，materialization order: base state → inserts → grid，keff 从 1.275 降至 1.249（物理正确：endplug/plenum 不再含 UO2）；(2) Exact spacer-grid frame geometry——`inner_side = sqrt(pitch² - A_cell)`，`frame_thickness = (pitch - inner_side)/2`，mass back-calculation 验证 8 bands 全部 < 1e-6 relative error，assembler 支持 `square_frame`/`box` region_kind（8 XPlane/YPlane surfaces + boolean expression），`CellLayerPatch` 扩展 `outer_side_cm`/`inner_side_cm`；(3) Host-path equivalence validation——新增 `host_path_validation.py` 模块，`validate_replacement_host_equivalence()` 检查 replacement universe 保留 host guide-tube wall（outermost wall layer by r_max），VERA4 Pyrex/thimble/RCCA 全部通过；(4) Cross-process deterministic identity certification——新增 `scripts/certify_fullcore_deterministic_identity.py`，PYTHONHASHSEED=1 vs 98765 跨进程 plan digest / object IDs / counts 全部一致（10/10 checks），SHA-256 content hash 不依赖 Python `hash()`，`_compute_pin_state_hash` 纳入 base_role/bpath_ids。Fail-closed 加固：missing grid density/mass 报告 error 而非默认值。非-openmc `1835 passed`，OpenMC `388 passed`，benchmark `21/21`，VERA4 acceptance `38/38`（keff=1.24919±0.00737），cross-process identity `10/10`。**P2_FULLCORE_BASE_PATH_STATE_MATERIALIZATION_READY**。**P2_FULLCORE_HOST_PATH_PRESERVATION_VALIDATED**。**P2_FULLCORE_DETERMINISTIC_IDENTITY_CERTIFIED**。

### 2026-07-16

- **P2-FULLCORE-2D-A 补完：Spacer Grid 物理化、Canonical State Hash、Structured Issues**：materializer 重写完成三项增强：(1) Spacer grid assembly-instance materialization——`AssemblyGridState`/`GridFrameDerivationReport` typed results，mass conservation derivation（`A_cell = grid_mass / (density × 289 × grid_height)`，frame thickness = `A_cell / (4 × pitch)`），grid state 计入 pin state hash，8 bands × 9 assemblies = 72 instances tracking；(2) Canonical state hashing——derived pin lattice/universe/cell ID 使用 SHA-256 content hash（`assembly_lattice__<type>__<hash16>`），不依赖 `seg_counter`；core lattice dedup by content hash（相同 core pattern 复用同一 lattice），VERA4 从 19 lattices 降至 16（3 个重复 core 状态被合并）；`state_reuse_report` 返回 pin/core reuse ratio；(3) Structured fail-closed issues——`MaterializationIssue` typed dataclass 替代 `dict[str,str]`，`ConcreteAxialStateResult.has_errors` property，assembler 传播 materialization issues 到 `PlanAssemblyResult`；validation：`base_lattice_missing`、`whole_plane_fill_missing`、`whole_plane_ref_missing`、`coordinate_out_of_bounds`。非-openmc `1803 passed`，OpenMC `388 passed`，benchmark `21/21`，VERA4 acceptance `38/38`（keff=1.27484±0.00741）。

- **P2-FULLCORE-2D-A VERA4 Deterministic Base-Case Fidelity Closure**：新增 axial segment `fill_mode` 契约（`detailed_core` / `whole_plane_material` / `whole_plane_universe` / `void`），`compile_global_axial_segments` 接受 `base_axial_layers` 并将每个 segment 映射到唯一基础 axial layer、自动分类 fill mode、domain clipping（breakpoints 裁剪到 domain 范围）；materializer 重构支持 whole-plane segment（material/universe/void fill 不生成 derived pin/core lattices）；坐标规范化（materializer 不再假设 insert coordinates 已经是 0-based，通过 `coordinate_convention.index_base` 自动转换）；0.0 truthiness bug 修复（`z_min_cm=0.0` 不再被当作 None）；完整 VERA4 fixture（12 base axial layers: -55→463.937 cm，含 lower/upper moderator buffer、core plate、nozzle、shoulder gap、fuel endplug、active fuel、plenum）；nozzle/core-plate 混合材料（SS304+coolant vol% mixture）；完整 fuel-path universes（active_r1/r2、endplug、plenum、water_pin、guide_tube with wall、instrument_tube with wall）；exact Pyrex coordinates（20 per edge assembly，1-based source convention）；thimble plugs（24 per corner + 4 per edge = 112 total）；formal RCCA multi-segment profile（AIC/B4C/plenum/endplug via `LocalizedInsertProfilesPatch`，anchor=257.9，control_state="base"）；assembly-scoped spacer grid overlays（8 bands，Inconel+Zircaloy）；新增 `campaign_eval/vera4_base_acceptance.py`（5-level acceptance: facts/plan/geometry/XML/runtime，38 checks）；新增 `scripts/vera4_base_fidelity.py`（full pipeline: assemble → render → XML → geometry debug → smoke → acceptance）；VERA4 deterministic base-case acceptance **38/38** checks passed（model.py 142KB → 5 XML → geometry loads → transport smoke keff=1.27484±0.00741 → zero lost）。非-openmc `1783 passed`，OpenMC `388 passed`，benchmark `21/21`。**VERA4_DETERMINISTIC_BASE_CASE_ACCEPTANCE_PASSED**。**VERA4_DETERMINISTIC_BASE_CASE_SMOKE_PASSED**。

- **P2-FULLCORE-2C-B Concrete Insert State Materialization and VERA4 Physical Closure**：新增 `axial_state_materializer.py`（per-segment derived pin lattices with insert overrides、per-segment wrapper universes/cells、per-segment core lattices、`CoreSpec.axial_layers` with segment-specific lattice fills）；production assembler 消费 `LocalizedInsertProfilesPatch`（resolved profiles → global axial segments → concrete materialization → axial layers）；VERA4 fixture 增加真实物理结构（pyrex_rod/rcca_absorber replacement universes with cylinder regions + background coolant、pyrex_glass/rcca_absorber_mat materials with full composition）；修复 core axial root cell 径向 boundary 使用 `core.boundary`（reflective）而非 `assembly.boundary`（transmission）；core lattice outer 保留为 precision safety net（不再清零，不再 block renderer）；renderer 移除 `core.lattice_outer_unreachable` error check。VERA4 deterministic physical closure 21/21 checks passed（patches → plan → CoreRenderer runnable → model.py 64KB → 5 XML files → geometry loads → **transport smoke**）。非-openmc `1708 passed`，OpenMC `388 passed`，benchmark `21/21`。**VERA4_DETERMINISTIC_PHYSICAL_CLOSURE_PASSED**。

- **P2-FULLCORE-2C-A Profile Registry, Anchor Resolver, CoreRenderer Export, and VERA4 Geometry Closure**：新增 `LocalizedInsertProfilesPatch` 正式 patch 容器（profile registry）；新增 `localized_insert_profiles.py` 模块含 anchor 解析器（`resolve_profile_anchor`/`resolve_profile_absolute_segments`，支持 bottom/top/center/absolute 四种 anchor_kind）、typed result（`ResolvedLocalizedInsertProfile`/`ResolvedSegment`）、profile validators（`validate_profile_registry`/`validate_profile_segments`/`validate_profile_references`）；executor 扩展 `_DEFAULT_ORDER`/`_DEPENDENCIES`/`_PATCH_DEPENDENTS` 含 localized_insert_profiles；patch prompt 新增 reactor-neutral profile 规则；`PatchGenerationContext`/`PatchValidationContext` 扩展 `known_insert_profile_ids`/`insert_profile_summaries`/`movable_insert_facts`；`compile_global_axial_segments` 重构接受 resolved profiles 并将 profile segment boundaries 注入 global breakpoints；core lattice outer_universe_id 在 reflective/vacuum boundary 时自动清零（消除 dead geometry renderer flag）。VERA4 deterministic geometry closure 15/15 checks passed（patches → plan → CoreRenderer runnable → model.py 36KB → 5 XML files → geometry loads OK）。非-openmc `1708 passed`，OpenMC `388 passed`，benchmark `21/21`。**VERA4_DETERMINISTIC_GEOMETRY_DEBUG_PASSED**。

### 2026-07-16

- **P2-FULLCORE-2B Moderator Outer Universe + Insert Axial Profile + CoreRenderer Export Readiness**：修复 pin lattice `outer_universe_id` 从使用 `default_universe_id`（燃料 pin）改为专用 `moderator_outer` universe（含 `UniverseSpec`+`CellSpec` fill_type=material fill_id=coolant）；新增 `ensure_moderator_outer_universe()` helper；assembly gap 计算（pin_pitch×lattice_size vs assembly_pitch）；新增 `LocalizedInsertAxialSegmentPatchItem`/`LocalizedInsertAxialProfilePatchItem` schema 支持多段 RCCA axial profile（absorber+plenum+end structure）；`LocalizedInsertIntentPatchItem` 新增 `axial_profile_id`/`anchor_z_cm`/`control_state_id` 字段支持参数化棒位；CoreRenderer export 验证（wrapper cells reference real lattices、core lattice universe_pattern 引用 valid universes、plan serializable）。非-openmc `1679 passed`，OpenMC `388 passed`，benchmark `21/21`。**P2_FULLCORE_AXIAL_PROFILE_SCHEMA_READY**。

- **P2-FULLCORE-2A Production Hierarchical Plan Integration + Full-Core Renderer Readiness**：将 `AssemblyCatalogPatch`/`CoreLayoutPatch` 接入 production `assemble_simulation_plan_from_patches`；multi-assembly 路径生成 `ComplexModelSpec(kind="core")` 含 per-type pin lattices + assembly wrapper universes/cells + core lattice；`_expand_assembly_pin_map` 不再将 localized inserts 写入 base lattice（Pyrex/thimble/RCCA 仅通过 axial loadings 应用）；assembly boundary 改为 `transmission`（内部组件不得 reflective）；core lattice 居中原点 (`center_cm=(0,0)`, `lower_left_cm=(-w/2,-w/2)`）；新增 typed `HierarchicalCoreAssemblyResult`（替代 raw dict）；新增 `compile_global_axial_segments`（从 facts/inserts/spacer grids 编译无 gap 无 overlap 全局轴向分段）；新增 3 个测试文件（production assembler integration / base lattice purity / global axial segments / boundary gap placement）。VERA4 deterministic production render 14/14 checks passed（kind=core、3 pin lattices、3 wrapper universes、core_total_fuel=2376、transmission boundary、centered core lattice）。非-openmc `1659 passed`，OpenMC `388 passed`，benchmark `21/21`。**P2_FULLCORE_PRODUCTION_PLAN_ASSEMBLED**。

- **P2-FULLCORE-1 Scope-Aware Count Contract, Hierarchical Assembly Catalog, Core Layout Patch, and Incremental Full-Core Planning**：新增 `ModelScope`/`CountScope`/`ScopedExpectedCount` schema；FactsPatch 扩展 `model_scope`/`assembly_count`/`core_lattice_size`/`assembly_type_counts`/`scoped_expected_counts` 等字段；新增 `scoped_counts.py`（`normalize_scoped_counts`/`compute_assembly_pin_counts`/`aggregate_core_counts`/`validate_count_scope_compatibility`/`compare_scoped_expected_counts`/`derive_homogeneous_local_counts_if_proven`）；新增 `AssemblyCatalogPatch`/`AssemblyTypePatchItem`/`AssemblyPinMapPatchItem`/`CoreLayoutPatch` 四个 patch schema；新增 catalog/layout validator + cross-validation；新增 `hierarchical_assembler.py`（`lift_single_pin_map_to_catalog`/`assemble_assembly_templates`/`assemble_core_lattice`/`build_hierarchical_core_plan`）；executor `_DEFAULT_ORDER`/`_DEPENDENCIES`/`_PATCH_DEPENDENTS` 扩展 assembly_catalog/core_layout；`required_patch_types_for_state` 区分 single/multi path；retry routing 新增 scoped count error codes；prompts 新增 assembly_catalog/core_layout rules + facts prompt 扩展 scoped counts；PatchGenerationContext/PatchValidationContext 扩展 multi-assembly fields。18 新测试文件（104 tests）覆盖 scope schema/validation/aggregation/homogeneous derivation/heterogeneous no-division/catalog patch/layout patch/prompts/dependencies/validators/cross-validation/hierarchical assembler/template reuse/localized insert scope/single-assembly compat/VERA4 regression。非-openmc `1626 passed`，OpenMC `388 passed`，benchmark `21/21`。VERA4 planning diagnostic：3 assembly types (corner/edge/center_rcca)、3×3 layout、core_total fuel=2376/guide_tube=216/instrument_tube=9、no cross-scope mismatch（264 vs 2376 不再比较）。**P2_FULLCORE_IR_READY_RENDERER_NOT_YET_VALIDATED**。P1 certified baseline SHA=6da53e2；current HEAD P2 development uncircified。

- **P1-RUNTIME-POSTFREEZE-2 Localized Insert Intent Contract + Thimble Closure + Full Requalification**：新增 `LocalizedInsertIntentPatchItem` schema 将有限轴向插入件（Pyrex/thimble/absorber）从基础 lattice 路径语义中分离；PinMapPatch 新增 `localized_insert_intents` 字段；`expand_pin_map` 不再读取 `pyrex_rod_coords`/`thimble_plug_coords` 作为基础 lattice 赋值；新增 `localized_insert_derivation.py` 通用确定性 insert loading 派生管线（替代 pyrex-only `_normalize_axial_insert_pin_map`）；assembler 集成 `derive_localized_insert_loadings` 处理所有 insert kind；validator 不再原地修改 patch（改为仅报告 overlap）；axial_overlays `total_mass_missing` 降级为 warning（renderer 可从 frame geometry 计算）；acceptance helper 的 `collect_loading_override_counts` 同时计数 `coordinate_override` transformation。Qualification N=10: **10/10 FIRST_PASS_SUCCESS, 10/10 full VERA3B acceptance (100%), 10/10 real LLM, 10/10 real OpenMC, 0 lost particles, 0 unsafe**。Transport seed stability: **3/3 PASSED (mean keff=1.00554, max z=1.01, effective batches=20/particles=10000)**。非-openmc `1522 passed`，OpenMC `388 passed`，benchmark `21/21`，Lane A `75 passed`。**thimble_loading_missing CLOSED。P1_RUNTIME_STAGE_COMPLETE_CURRENT_HEAD**。

### 2026-07-15

- **P1-RUNTIME-POSTFREEZE-1 Material Semantics Closure + Current-HEAD Requalification**：新增 `CompositionValueBasis`/`NormalizationStatus` 枚举 + `material_semantics.py`/`material_normalization.py`/`material_validation.py` 三模块；executor 删除 `_reconcile_uo2_oxygen_scale`/`_reconcile_borated_water_boron` 隐藏修正，renderer 忠实渲染 normalized material；assembler 集成 `normalize_material_semantics` 在 assembly 后自动归一化（declared basis → deterministic transform → ambiguous blocked）；material patch prompt 扩展 `composition_basis` 语义说明 + reactor-neutral 示例（UO2 stoichiometric + borated water ppm）；axial overlay prompt 从 `homogenized_open_region` 改为 `mass_conserving_outer_frame` 默认。新增 `certification_identity.py`（CertificationIdentity + PhysicsContractIdentity + `check_certification_stale` SHA/contract hash mismatch detection）；transport seed evaluator 修复 `batches`/`particles` 实际写入 settings.xml + `canonical_settings_hash_excluding_seed` + 删除 `-s` 参数（OMP_NUM_THREADS 控制线程）；acceptance 模块改为 repo-root sys.path + `FullAcceptanceLoadError` 不再静默 fallback；CLI `profile=qualification` 自动强制 full acceptance。14 新测试文件（1522→1522 non-openmc passed）。Qualification N=10: 10/10 FIRST_PASS_SUCCESS, 10/10 real LLM verified, 10/10 real OpenMC verified, 0 lost particles, material normalization 全部正确（fuel stoichiometric_ratio→atom_fraction, coolant ppm_by_weight→atom_fraction）。VERA3 acceptance: 0/10 passed (每 run 仅剩 2 个 `thimble_loading_missing`，spacer grid 已修复)。Transport seed stability: 3/3 PASSED (mean keff=1.0217, max z=1.31, geometry/materials/canonical-settings hashes 全 match, effective batches=20/particles=10000 正确)。

### 2026-07-14

- **keff 材料渲染 bug 修复**：发现两个系统性 production executor bug 导致 LLM 生成模型 keff≈0.65（预期≈0.98）。Bug 1：燃料 O16 用分子比 ao=2.0（2 O/U）与 enrichment ao=2.619（per-100-U）混用，OpenMC 归一化后 O/U=0.02 而非 2.0，燃料实际变为铀金属。Bug 2：硼水 B10 用 ppm 值 ao=0.001066（1066 ppm）直接作为 atom fraction，实际过量 9×。修复：executor 新增 `_reconcile_uo2_oxygen_scale()`（检测 U percents sum≈100 且 O16≈2 → 乘以 100）和 `_reconcile_borated_water_boron()`（检测 B10 frac>5e-4 的水类材料 → 从 ppm 转换为正确 atom fraction）。修复后 keff 从 0.65 提升到 0.985±0.006，达到 VERA3B 预期范围。11 新测试。非-openmc `1455 passed`。

- **P1-RUNTIME-TRUTH-3 Qualification N=10 + Transport Seed Stability + Final Gate**：强化 LLM 证据 9-point gate（`LLMCallRecorder` 统一记录 planning/diag/proposer/supervisor 四角色调用，拦截 `__call__` + `generate_patch_json`）；3-stage OpenMC 证据（export/geometry-debug/smoke 各自独立 backend 标记 + returncode + hash）；实际 reference/few-shot/monolithic provenance 从 workflow state 提取替代硬编码空列表；campaign budget 强制（`LLMBudgetExhausted` → `SAFE_STOP_BUDGET`）；safe resume（config mismatch → `CAMPAIGN_RESUME_CONFIG_MISMATCH`）；扩展 qualification metrics（Wilson 95% CI、autonomous/bounded rates、success-only artifact/verification rates）；transport seed stability evaluator（3 seeds × pairwise z ≤ 5）；P1-RUNTIME final gate（10 gates 全 PASS）。Qualification N=10: 9/10 FIRST_PASS_SUCCESS (90%), real_llm_verified=100%, real_openmc_verified=100%, unsafe=0, artifact=100%。Transport seed stability: 3/3 PASSED (max z=0.61)。非-openmc `1444 passed`，OpenMC `388 passed`，benchmark `21/21`，Lane A `20/20`。**P1_RUNTIME_STAGE_COMPLETE**。

- **P1-RUNTIME-TRUTH-2 Real Lane B Executor + VERA3B Pilot N=3**：实现真实 Lane B campaign executor（`real_campaign.py`），每次 run 创建全新 DeepSeek client，通过 production `build_plan_graph` 完成 incremental planning → validation → render → real OpenMC → bounded runtime recovery。Pilot N=3 全部通过：3/3 FIRST_PASS_SUCCESS，每个 run 真实 OpenMC export/geometry-debug/smoke 成功，zero lost particles，VERA3B acceptance 通过。Safety: unsafe=0, fake_client=false, reference_patches=0, monolithic_fallback=false。修复 run classifier（runtime supervisor finish_success ≠ runtime_iters==0）。非-openmc `1349 passed`，benchmark `21/21`。

- **P1-RUNTIME-TRUTH-1 Runtime Evaluation Truthfulness Closure**：修正 promotion gate 语义（PARTIAL/PASSED/FAILED + new field tracking）；新增 source-strategy rendering contract（4 策略 active_fuel_box/assembly_box/manual/unknown 流入 assembler→renderer→settings.xml 完整链）；source_bounds_for_plan 重构为 strategy-aware；atom-density fuel detection（density_unit=sum 不再需要注入虚构 g/cm3）；删除 fixture 后处理密度补丁；F01 关闭为真实 OpenMC source recovery gate（manual non-fuel bounds → pre-flight → deterministic repair → candidate clone real smoke → commit → rerun success）；F05 重分类为 root-cause precedence injection gate（NO_PROGRESS）；删除 _REAL_OPENMC_CASES 硬编码集合，改为 case.requires_real_openmc 驱动；修复 5 个生产 bug（GraphState 缺 runtime_supervisor_result/runtime_user_cancelled、RuntimeFailureClass import ×3、repair_result_router 缺 runtime_supervisor edge、repaired plan capability re-assessment、ValidationReport 构造）。20/20 fault matrix PASSED，2 real-OpenMC (F00+F01)；非-openmc `1301 passed`，OpenMC `388 passed`，benchmark `21/21`。

- **P1-RUNTIME-R7/R8 VERA3B Fault-Injection Matrix + Runtime Stability Campaign**：将占位 harness 改造为 production graph-backed runner；新增 `runtime_faults.py`（20-case immutable fault registry + 隔离 source-state injectors）、`runtime_metrics.py`（matrix/campaign aggregation + promotion gates）、`runtime_campaign.py`（per-case tool/LLM injection 通过 `build_plan_graph(...)` 参数注入，不另写模拟状态机）；修复 5 个 R5/R6 遗留 production bug（`stable_json_hash`/`RuntimeFailureClass` import 缺失、`plots/` 目录未创建、`retry_same_plan` 不重置 validation_report、`GraphState` 缺 `accepted_plan_build_state` 字段）；Lane B CLI 有 confirmation gate（缺 key → `NOT_RUN_ENV`，有 key 无 `--confirm-real-campaign` → `CONFIRMATION_REQUIRED`）。18/20 evaluated cases 通过（F00 real OpenMC baseline + 17 injected），2 pending real-OpenMC（F01/F05 source rejection injection 需要 renderer source-binding 深入修改）。36 新测试；非-openmc `1254 passed`，OpenMC `388 passed`，benchmark `21/21`。

- **P1-RUNTIME-R5/R6 Post-Execution Runtime Supervisor + Bounded Recovery**：新增独立 `runtime_supervisor.py`/`runtime_supervisor_policy.py`/prompt（不复用 planning RunSupervisor action），包含 finish/deterministic/LLM/transient retry/human/stop 六种 action、RuntimeLoopBudget、Python allowlist/veto、runtime-only fingerprint 和 no-progress 保护；feature-flag graph 将执行结果进入 runtime supervisor，repair commit 后从 source patch 重组 plan 再 reexecute，transient 同计划仅重试一次；新增 runtime_loop iteration manifests 与 VERA3B fault-sequence harness（缺真实密钥时 `REAL_LLM_SKIPPED_ENV`）。复杂 geometry 继续 diagnose-only，不能扩大 provenance allowlist。16 新测试；R4 完整回归非-openmc `1252 passed`，OpenMC `388 passed`，benchmark `21/21`。

- **P1-RUNTIME-R4 LLM Runtime Diagnostician + Constrained Patch Proposer**：在 R2/R3 基础上新增 `runtime_diagnostician.py`（RuntimeDiagnosis + 10 种 RepairKind + evidence 分级 + deterministic validation，LLM 只能收窄权限不能扩大）；新增 `runtime_patch_proposer.py`（LLMRuntimeRepairProposal + 静态 RFC6902 validation，复用 `apply_json_patch_to_clone` + `is_protected_path` 三层安全网）；RuntimeRepairPolicy 扩展 LLM 字段（llm_diagnosis_supported / llm_proposal_supported / allowed_repair_kinds / max_mutating_operations=4）；graph 新增 `llm_runtime_diagnose → llm_runtime_propose` 路由（one-shot budget，fingerprint dedup，allow_fallback=False 不用 Fake 冒充真实 LLM）；新增 prompt builder 和 16 个 trace event。29 新测试覆盖 diagnosis validation、proposal static safety（protected/forbidden/budget/allowlist/root）、fake client safety、graph routing。非-openmc `1236 passed`，OpenMC `388 passed`，benchmark `21/21`。

- **P1-RUNTIME-R2/R3 Deterministic Runtime Repair + One-Shot Recovery**：修复 validate_plan 中 incremental regeneration 清空 plan_build_state 导致 valid patch 丢失的 bug（graph.py:2708）；新增 `runtime_repair_policy.py`（18 个 issue code 的 RuntimeRepairPolicy registry，映射 owner 到真实 patch type）；新增 `runtime_repair.py`（RuntimeRepairRequest/Proposal/Evaluation + source binding oracle + geometry diagnosis + clone-only evaluation + commit）；graph 新增 classify_runtime_feedback → deterministic_runtime_repair 路由（one-shot budget，fingerprint dedup，不消耗 planning retry_count）；新增 12 个 trace event 类型。source oracle 确定性切换 source_strategy→active_fuel_box + fissionable=true，不写入 benchmark 常数。23 新测试；非-openmc `1207 passed`，OpenMC `388 passed`（含此前 pre-existing fail 现已修复），benchmark `21/21`。

- **P1-RUNTIME-R0/R1 Runtime Feedback Contract + Geometry-Debug Stage**：新增 `runtime_feedback.py`（`RuntimeFailure`/`RuntimeFailureClass`/`RuntimeIterationRecord` + `classify_runtime_tool_results`/`normalize_runtime_error`/`compute_runtime_error_fingerprint`，根因优先级 source-rejection > cross-section > geometry > material > crash > unknown；fingerprint 去除时间戳/PID/绝对路径/hex 地址）；新增 `run_geometry_debug` 工具（隔离 `geometry_debug/` 子目录，timeout 不归为 overlap）；重构执行顺序 export→plots→geometry_debug→smoke，每阶段独立 trace event；error catalog 新增 `runtime.openmc_timeout`/`runtime.openmc_process_crash`；修复 `_runtime_issue` message=None bug 和 "No overlaps found" false positive。29 新测试；非-openmc `1187 passed`，OpenMC `384 passed`（1 pre-existing fail），benchmark `21/21`。

- **Incremental ID Canonicalization + Half-Pitch Water Cell**：axial patch generator 对唯一匹配的 universe ID 分隔符变体做保守规范化（如 `fuel_pin_endplug`→`fuel_pin_end_plug`），不为真正未知/歧义 ID 猜测替换；assembly capability 允许 moderator cylinder 半径恰好等于 square lattice half-pitch，仍拒绝超界半径。根因：真实 VERA3B LLM 产生 endplug 拼写变体，且 `water_cell r=0.63 cm` 被错误判为超出 `pitch/2=0.63 cm`，使可渲染 plan 被降为 skeleton。2 新测试；非-openmc `1159 passed`，benchmark `21/21`。

- **Structural Axial-Slab Material Guard**：axial patch validator 新增 reactor-neutral 规则：`lower_nozzle`/`upper_nozzle`/`core_plate` 等全截面结构层不得引用 role=`coolant`/`moderator` 的材料；materials/axial prompts 明确要求将输入给出的钢-冷却剂均匀化结构写为独立 material + `mixture_components`，不能只写 assumption 后填充纯水。根因：真实 VERA3B LLM patch 将 4 个 nozzle/core-plate slab 直接填为 borated water。2 新测试；fixture 重建确认 4 层使用 `lower_nozzle_3b`/`upper_nozzle_3b`/`core_plate_3b`；非-openmc `1157 passed`，benchmark `21/21`。

- **Incremental Axial-Universe Dependency Recovery**：当 `axial_layers` 的 lattice transformation 引用了未定义 replacement universe 时，executor 提取缺失 ID，保留有效 facts/materials，并仅失效/再生 `universes` 与其下游 pin-map/axial patches；下一轮 universe prompt 显式携带所需 profile ID。修复了此前全量 7-patch regeneration 反复消耗 retry budget 的依赖环；不启用 benchmark reference、gold few-shot 或 monolithic fallback。1 新测试；非-openmc `1155 passed`，benchmark `21/21`。

- **Concentric Pin/Tubing Closure（修复 VERA3B lost particles）**：assembler 现在按显式 `r_min_cm>0` 生成 annulus，即使 LLM 将 gap/clad 标成 `cylinder`；对有径向外边界、但未声明 background cell 的 repeated universe，确定性注入输入材料 role=`coolant`/`moderator` 的外部 moderator cell，避免 lattice pitch 内未定义区域。根因：最新 VERA3B LLM plan 将 gap/clad 渲染成嵌套实心圆柱且漏掉外部 coolant，粒子穿越 clad 外径 surface 3 后无法定位。真实 VERA3B 重建输运完成（keff=0.99345±0.00592，0 lost particles）。2 新测试；非-openmc `1154 passed`，benchmark `21/21`。

- **Renderer nuclide-name normalization（修复 B-10 transport abort）**：executor 新增 reactor-neutral `_normalize_nuclide_name()`，将 GND 连字符核素名（`B-10`/`U-235m`/`B-10.71c`）统一改写为 OpenMC HDF5 库约定（`B10`/`U235m`/`B10.71c`），在全部 5 个 `add_nuclide` 出口（2 live 构造 + 3 render emit）应用；纯格式化，不改成分/密度。根因：OpenMC 0.15.x 原样存储核素名，本机 `endfb-vii.1-hdf5` 库仅含无连字符条目，导致 VERA3B 硼水/Pyrex 的 `B-10` 在输运期 `Could not find nuclide` MPI_ABORT。真实 VERA3B 端到端：smoke exit=0、nuclide 错误消除、keff≈0.951。4 新测试；非-openmc `1150 passed`，OpenMC executor `49 passed`，benchmark `21/21`。

- **P0-FINAL + FC-MVP：工程基线闭合 + 全堆 MVP**：VERA3 reference Table P3-3 精确原子数密度（atom/barn-cm）替换全部存根材料；新增 `atom_density_barn_cm` composition_basis + `set_density('sum')` 支持；executor 扩展原子质量表（60+ 核素）用于 outer-frame density 反算。VERA3 3A/3B transport smoke 通过（3A keff=1.17547±0.005 Δ=0.025%，3B keff=0.99487±0.006 Δ=0.53%）。Reusable assembly universe（boundary=transmission, x/y bounds stripped）支持 core lattice 嵌套；3×3 full-core MVP transport smoke 通过（keff=0.99452±0.005, leakage=4.6%, 0 lost particles, assembly tally 9 entries center-peaked P/A=1.35）。非-openmc `1147 passed`，OpenMC `380 passed`，benchmark `21/21`。

- **P0-V4 Variant-Specific Nozzle and Core-Plate Homogenized Mixtures**：新增 `MaterialSpecPatch.mixture_components`/`variant_scope`/`derivation_method` 和 `ComplexMaterialSpec.mixture_component_ids`/`mixture_volume_fractions`/`is_mixture` schema 支持；executor 新增确定性 volume-fraction 展平（`_flatten_volume_mixture` 将 atom_frac/weight_frac 组件统一转为 weight_frac 混合，不依赖 `openmc.Material.mix_materials`）。3A/3B 各新增 3 个 variant-specific mixture（lower_nozzle f_SS304=0.27922 / upper_nozzle f_SS304=0.19147 / core_plate 50-50）；4 层 axial layer fill 从纯 SS304 迁移至 variant mixture；renderer/executor density+composition 校验跳过 mixture 材料。28 新测试；非-openmc `1117 passed`，OpenMC `380 passed`，benchmark `21/21`；3A/3B geometry 0 overlaps / 0 lost particles。

- **P0-V3 Spacer-Grid Mass-Conserving Outer-Frame Geometry**：新增 `mass_conserving_outer_frame` geometry_mode（Level 2 overlay）和 reactor-neutral `outer_frame_overlay.py` planner（`derive_mass_conserving_outer_frame` 确定性计算 frame area/thickness/mass conservation + clearance check）；executor 新增 outer-frame 发射（4 XPlane/YPlane + inner region + frame cell + 保留 background moderator 至 inner square）。3A/3B 8 个 spacer-grid 从 `homogenized_open_region` 迁移至 `mass_conserving_outer_frame`（end 1017g Inconel-718 / mid 875g Zircaloy-4 / 289 cells）；新增 schema 字段 `total_mass_g`, `cell_count`, `pitch_cm`, `material_density_source`, `mass_tolerance_rel`。38 新测试；非-openmc `1117 passed`，OpenMC `380 passed`，benchmark `21/21`；3A/3B geometry 0 overlaps / 0 lost particles。

- **P0-V2 VERA3B Pyrex Upper-Gas Axial Profile**：新增 `pyrex_upper_gas_inner_profile` universe（SS304 内/外管保留、氦气腔替代毒物+间隙、水隙背景保留）和 `pyrex_upper_gas_loading`（16 坐标复用毒物段）；6 层轴向层（376.441–397.510 cm）更新为含 upper-gas 的多 loading 组合；geometry contract v1→v2（双 axial profile + conflict resolved）。修复多 loading 物化时共享 `derived_lattice_id` 导致 lattice ID 碰撞的 reactor-neutral 问题（`materialize_axial_lattice_transformations` 追加唯一后缀）。41 新测试；非-openmc `1079 passed`，OpenMC `380 passed`，benchmark `21/21`。

- **P0-V1 Fuel Helium Gap & Upper Plenum Correction**：新增燃料棒氦气间隙 (0.4096–0.418 cm) 到 fuel_pin universe；修正 plenum 气体 r_max (0.4096→0.418) 和 clad r_min (0.4096→0.418)；新增 reactor-neutral `radial_profile_validation.py` validator（间隙/重叠/背景检测）；集成到 patch validator。修复 7 个 P0-D5B 预存测试失败（frozen 回归 fixture）。44 新测试；非-openmc `1038 passed`，OpenMC `380 passed`，benchmark `21/21`。

### 2026-07-18

- **P2-PLAN-CLOSED-LOOP-PHASE-7A**：Real Controlled Five-Gate Canary 离线 harness（reactor-neutral `RealCampaignCaseSpec`/`builtin_case_registry` 仅在 case preset/CLI/fixture/report 出现，生产代码无 VERA3/VERA4 分支；`RealCampaignClientBundle` 扩展 `plan_reviewer_client`+`plan_repair_client`，由 `_create_client_bundle(plan_reviewer_enabled, plan_repair_enabled)` 构造并通过 `LLMCallRecorder` 记录；`detect_provider_environment(model)` 通过 `_client_for_model` resolver + `OpenAICompatibleChatClient.api_key_env` class attribute 探测 provider key env — 不硬编码 `DEEPSEEK_API_KEY`，未知 provider 返回空 env → `BLOCKED_BY_LLM_ENVIRONMENT`；`make_five_gate_controlled_policy()` 显式五 Gate controlled；`estimate_real_campaign_llm_budget` 估算 patch/manifest/fragment/gate-review/plan-repair/runtime 8 类预算；`CampaignResumeFingerprint` 18 字段绑定 git/input/req/human-answer/model/provider/reasoning/output/policy-hash/gates/modes/universes-mode/fragment-budget/material/runtime/cross-sections fingerprint，不匹配 → `CONFIG_MISMATCH`；`validate_real_canary_truthfulness` 覆盖 fake/fallback/reference/gold/monolithic/render-before-final-gate/partial-fragment/missing-reviewer/gate-auto-accepted/stale-plan/reasoning-persisted；stages `planning|render-compile|openmc-smoke`；artifact writers 含 five_gate_status/timeline/hashes、plan_reviewer_calls、llm_budget、truthfulness_evidence、environment_evidence、human_answer_provenance；CLI `scripts/evaluate_plan_closed_loop_real_canary.py` 含 `--case`/`--input`/`--operating-state`/`--stage`/`--universes-generation-mode`/`--strict-structured-patch-output` 等。复用 runtime_repair/supervisor/diagnostician/proposer 不重设计；不修改 contract 版本（仍 0.8）；不动 `_DEFAULT_EXECUTABLE_PLAN_GATES`。非-openmc `2593 passed`（0 failures），Phase-7A targeted `99 passed`，VERA3/VERA4 offline `BLOCKED_BY_LLM_ENVIRONMENT`/`BLOCKED_BY_OPENMC_ENVIRONMENT` 路径全部通过，fake benchmark `21/21`。

- **P0-LARGE-STRUCTURED-PATCH-GENERATION-CLOSURE**：大型结构化 Patch 分片生成（`PatchLLMResponse` typed telemetry with finish_reason/token usage/reasoning fingerprint、`normalize_patch_llm_response()` 统一 str/PatchLLMResponse、`strict_structured_output` fail-closed policy、`generate_patch_json_with_meta()` 返回 telemetry、`PatchGenerationAttempt` 新增 finish_reason/prompt_tokens/completion_tokens/reasoning_tokens/structured_fallback_reasons、`universe_fragment_generation.py` 含 requirement extraction/manifest/fragment model/deterministic merge/strategy switching/checkpoint session、`universe_patch_pipeline.py` 入口 `generate_universes_patch()` orchestrating auto→monolithic→fragmented truncation-aware strategy switching、executor 对 universes 调用 pipeline、CLI `--universes-generation-mode`/`--universe-fragment-max-tokens`/`--large-patch-safe-output-ratio`/`--strict-structured-patch-output`）。思考模式不关闭；可靠性来自缩小每次结构化输出而非更大 token 预算；截断后自动切 fragmented 不再请求同一个完整 Patch；fragment checkpoint/resume 跳过已完成 fragment；最终输出标准 UniversesPatch。非-openmc `2494 passed`（0 failures），targeted `61 passed`，VERA4 11-universe offline `全部通过`，fake benchmark `21/21`。

- **P2-PLAN-CLOSED-LOOP-PHASE-6**：Final / Assembled Plan Review Gate（contract 0.7→0.8 迁移、`AssembledPlanBindingView` 含 typed object graph/root selection/reachability/renderer capability matrix/static source feasibility/plot coverage/execution check、`AssembledPlanContractMatrix` 七行类型 root_selection/root_reachability/reference_integrity/renderer_capability/static_source_feasibility/plot_coverage/execution_check、`run_assembled_plan_preflight` 确定性 issue codes 覆盖 root/reachability/renderer/source/plot/exec、`AssembledPlanEvidencePack` typed evidence refs G/R/CM/SF/EC/D、独立 Critic `run_assembled_plan_review` + coverage + unknown-ref/owner-action/runtime-claim rejection、`assembled_plan_issue_policy.py` Python owner registry 路由 Facts/Materials/Universes/Placement/Axial-owned、executor `_run_assembled_plan_gate` controlled barrier requires all upstream gates accepted + Phase-3B retry 集成 + input-hash invalidation + reopen、CLI `--assembled-plan-review-mode`）。复用 assemble_simulation_plan_from_patches/validate_simulation_plan/choose_renderer/collect_active_dependencies/assembly3d_guard，不复制底层算法。非-openmc `2433 passed`（0 failures），Phase-6 targeted `63 passed`，VERA3/VERA4 offline `全部通过`，fake benchmark `21/21`。

- **P2-PLAN-CLOSED-LOOP-PHASE-5**：Axial Geometry Review Gate（contract 0.6→0.7 迁移、`AxialGeometryBindingView` + 9-kind `AxialGeometryContractMatrix`、`derive_axial_geometry_segments` 有限分段、`run_axial_geometry_preflight` 确定性 issue codes 覆盖 domain/fill/loading/overlay/through-path/localized-insert、`AxialGeometryEvidencePack` typed evidence refs F/M/P/B/A/L/O/I/T/G/D、独立 Critic `run_axial_geometry_review` + coverage + unknown-ref/owner-action/root-reachability/runtime rejection、`axial_geometry_issue_policy.py` Python owner registry 路由 Facts/Materials/Universes/Placement/Axial-owned、executor `_run_axial_geometry_gate` controlled barrier requires Facts+MU+Placement accepted + Phase-3B retry 集成 + input-hash invalidation + reopen、CLI `--axial-geometry-review-mode`）。复用 validators/axial_overlay/compute_axial_segments/assembly3d_guard/material_execution_readiness，不复制底层算法。非-openmc `2368 passed`（0 failures），Phase-5 targeted `103 passed`，VERA3/VERA4 offline `全部通过`。

- **P2-PLAN-CLOSED-LOOP-PHASE-4**：Material–Universe Review Gate（contract 0.5→0.6 迁移、`MaterialUniverseBindingView`/`MaterialUniverseContractMatrix` 四行类型、`run_material_universe_preflight` 确定性 40+ issue codes、`MaterialUniverseEvidencePack` typed evidence refs F/M/U/C/V/D、独立 Critic `run_material_universe_review` + coverage 完整性校验 + unknown-ref/owner-action/root-reachability rejection、`material_universe_issue_policy.py` Python owner registry、executor `_run_material_universe_gate` controlled barrier + Phase-3B retry 集成 + input-hash invalidation + reopen）。Gate 验证 Facts→Materials→Universes→cell 静态绑定、fuel-variant identity/reachability/collapse、role compatibility（reactor-neutral）、compound species evidence。role 兼容性区分 pass/fail/unresolved（custom role 不直接报 error）。非-openmc `2253 passed`（4 pre-existing failures 不变），Phase-4 targeted `64 passed`，VERA3/VERA4 offline `全部通过`，fake benchmark `21/21`。

- **P2-PLAN-CLOSED-LOOP-PHASE-3B**：typed retry request fidelity（5 builders + idempotent registration + lifecycle states）、scope-aware owner policy（pin_map+assembly_catalog 永不并存、unknown scope fail-closed）、`RetryCandidateProducerRegistry`（Facts/Materials/Universes/Placement/TaskPlan/Axial dedicated producers + `RetryPatchGenerationContext` retry-aware prompt）、real callable `run_owner_acceptance_checks`（25+ checks：schema/consistency/density/required-ids/near-miss/fuel-variant 等）、bounded `execute_plan_retry_loop`（多轮 + post-replay `reclassify_retry_outcome` + terminal removal + budget/cycle/no-progress enforcement）、`resume_incremental_from_patch`（非递归 depth-guarded downstream resume）、gate invalidation/replay 分离（`plan_retry_gate_invalidation_counts` vs `plan_retry_gate_replay_attempt_counts` vs `plan_retry_gate_replay_success_counts`）、Graph 新增 `execute_plan_retry` + `resume_plan_retry` 节点与 retry-aware routers、`retry_human.py`（typed human question + fingerprint-validated answer）、`retry_artifacts.py`。材料 readiness 路径现在同时注册 typed `ExecutablePlanRetryRequest`。非-openmc `2184 passed`（4 pre-existing failures unrelated），targeted Phase-3B `81 passed`，VERA4 offline mutations `6/6 PASS`，fake benchmark `21/21`。

### 2026-07-17

- **P2-FULLCORE-2D-B-MATERIAL-SPECIES-SEMANTIC-CLOSURE**：新增 typed `compound_components`、独立 `material_species` resolver、质量守恒/重复 species 合并、fissile compound fail-closed、legacy formula 审计转换、静态 patch gate、materials-only retry 与运行前 cross_sections 预检；所有普通/complex/mixture material emission 统一通过 canonical resolver。VERA4 Pyrex 现在以 `B2O3 12.5 wt% + SiO2 87.5 wt%` source 表示，确定性解析为自然 `B/Si/O`，并输出 `material_species_resolution_report.json`。验证：VERA4 A–G/geometry/XML/smoke 55/55（keff=1.14309±0.01647、0 lost），VERA3 3B XML 回归 0 error，fake benchmark 21/21。

- **P2-FULLCORE-2D-B-AXIAL-OVERLAY-SEMANTIC-CLOSURE**：axial-overlay `through_path_preserved` 区分字段缺失（确定性派生 True + 审计）与明确 false（`mode_semantic_contradiction` error）；issue-scoped compact retry 携带上次 parsed patch + 允许/锁定字段列表；retry drift gate 检测 z_min/material/total_mass 等锁定字段变化（`patch_retry.unexpected_semantic_drift`）；DS 模型 `patch_type` 自动注入（特征键匹配）；canary status 门控区分 fuel_variant_subcanary 与 full planning canary；post-decoration fuel variant identity 动态检查。VERA4 real-LLM (ds:deepseek-v4-flash) 全流程首次通过：8/8 patches valid、assembly ok、model.py compile + XML export + Geometry.from_xml + smoke test 全部 PASS。Level F (grid 11/11) + Level G (fuel 5/5) 全通。非-openmc `2016 passed`，OpenMC `388 passed`，benchmark `21/21`。

### 2026-07-12

- **P0-D5B Early Lattice-Loading Validation & Deterministic Grid Migration**：新增共享 `lattice_loading_validation.py`（`lattice_loading_structural_issues()` 在 validate_plan 阶段直接发现 `lattice_transform.replacement_universe_missing` / `source_universe_missing` / `loading_ref_missing`，不再推迟到 renderer materialization）；renderer `_axial_assembly_modeling_errors` 同步调用；`_probe_axial_materialization_blockers` 改为 defensive assertion（按 code 去重，不重复注入已发现 issue）。patch-level 新增 transformation cross-reference validation（replacement/source universe 存在性、cell-id-as-universe 误用、spacer_grid_transformation_misuse）；`PatchValidationContext` 扩展 `known_cell_ids` / `cell_owner_universe_ids` / `known_overlay_summaries` / `has_spacer_grids`。新增 `grid_loading_repair.py`（证据层级诊断 + Strategy A-D 确定性修复：优先移除冗余 grid transformation + 清理 layer loading_ids 引用，保留非 grid loadings；不创建 solid grid universe、不修改 materials/facts/pin_map）。graph repair pipeline 新增 lattice-loading/grid migration oracle（shoulder-gap oracle 之后、LLM 之前）。真实 VERA3B DeepSeek smoke：retry_count=0、grid_cell 引用消除、pin counts 264/24/1、capability issues=[]、model.py 生成；VERA3A 回归 retry_count=0、issues=[]。修复 3 个既有测试失败（fixture 缺 water_cell universe、test_resume 用不兼容 minimal universes）。非-openmc `994 passed, 1 skipped`；新增 6 个测试文件 `37 passed`。

- **P0-D5 Early Structural Validation & Deterministic Shoulder-Gap Repair**：统一 validator/renderer 共享 `assembly3d_structural_issues()`，使 `component_profile_as_material_slab` 在 validate_plan 阶段直接产生（不再推迟到 renderer capability）；新增 shoulder_gap role、patch-level early validation、forbidden role-edit ownership policy、确定性 multi-patch repair bundle（universes + axial_layers）+ clone-only acceptance + through-path preservation check。graph repair pipeline 新增 component-profile oracle（在 LLM repair 之前，支持多 layer 一次修复）。真实 VERA3 3A DeepSeek smoke：retry_count=0、pin counts 264/24/1、model.py 生成、capability issues=[]；6 个新测试文件 `36 passed`。

- **P0-D4 Deterministic Pin-Map Repair Oracle**：assembler 现在始终尊重 `PinMapPatch.default_universe_id`，不会被同 kind 的轴向 profile 覆盖，并为缺失显式默认产生结构化 assembly error。新增通用 pin-map usage/delta diagnosis、唯一等量 default 修复 oracle、clone-only preflight、severity-aware issue delta 与 enriched LLM semantic context。真实 VERA3 3A 本地 smoke 直接恢复 `fuel_pin=264, guide_tube=24, instrument_tube=1`，未调用 repair LLM；后续仅剩独立 shoulder-gap 结构问题。验证：新增/相关 targeted `32 passed`，OpenMC gate、full pytest、fake benchmark 均通过。

- **P0-D3 Real-LLM Patch Repair Proposal Contract**：将 DeepSeek repair 输出分为可兼容的 model envelope 与系统绑定的内部 proposal；缺失 rationale/confidence 使用可追溯保守默认，operations 仍严格校验。adapter 现在真实发送 JSON Schema，provider 拒绝后显式回退 json_object 并写入 raw/normalization/normalized artifacts。真实 VERA3 3A 两个候选均完成 clone parse/assembly/full validation 后因无改善拒绝，不再发生 metadata schema rejection。验证：contract targeted `24 passed`，full `1248 passed, 3 skipped`。

### 2026-07-11

- **P0-D2 Real-LLM Validation Repair Smoke**：修复 real-model validation repair 延迟构造 patch adapter、优先使用 adapter JSON mode，并为 schema-invalid proposal 写入 proposal/evaluation artifact。真实 DeepSeek VERA3 3A 两次局部 RFC6902 proposal 均因缺必填元数据被严格拒绝；仅一次 targeted regeneration、global retry=1，未出现三次盲重试。验证：repair targeted `13 passed`。

### 2026-07-10

- **P0-D1 Validation-Driven Incremental Patch Repair**：plan-level issue 现在先进入 patch-relative RFC6902 repair request（稳定 fingerprint、issue ownership policy、protected-path denylist、clone-only assemble/full validation）；accepted repair 不消耗 graph retry，重复/无改善候选立即停止并退回一次 targeted regeneration。VERA3 3A 历史三轮 `lattice.pin_count_mismatch` 已脱敏固化为回归 fixture/diagnosis。验证：新增 repair targeted `9 passed`，incremental/trace 回归 `44 passed`。
- **Incremental plan-level 定点修复**：validate_plan 不再对所有 incremental validation failure 清空 `PlanBuildState`；新增 issue→patch root 路由、patch/downstream invalidation 与 targeted repair prompt，优先重做命中的 patch 集。验证：incremental executor + graph integration `30 passed`。
- **VERA3 Geometry Step 1**：新增 test-only canonical geometry contract，分离 assembly zones、fuel/guide/instrument/Pyrex/thimble component profiles 与 3B finite loading；旧 reference 标为 legacy。升级 acceptance helper（base/loading/active union/continuity）和 deterministic diagnostic oracle，修复 12-layer 与 full-height Pyrex/plug 集成断言。当前 fixture 明确诊断 whole-layer end-plug/plenum、Pyrex helium-gap/radial-stack 和 missing thimble loading，不作为 gold/few-shot。验证：targeted `59 passed, 2 skipped`，OpenMC gate `356 passed, 2 skipped`，full `1105 passed, 3 skipped`。
- **P0-B LLM Patch Repair Proposer**：新增受控 repair proposal schema、issue→path allowlist、protected path policy、JSON Patch clone executor、before/after deterministic validation、accept/reject/unsafe 判定、prompt/fake client/fallback，并接入 workflow trace/artifacts 与 benchmark metrics/CLI。默认关闭/proposal-only，不运行 OpenMC、不修改科学事实；validate-only 只作用于 clone。验证：新增 repair proposal tests `19 passed`，相关旧测试 `27 passed, 9 skipped`，no-OpenMC 全量 `641 passed, 18 skipped, 10 deselected`，repair fake benchmark `16/16 pass_rate=100%`。
- **P0-A LLM Semantic Plan Auditor**：新增只读 semantic audit schema/input builder/prompt/client/fake/fallback，并接入 workflow trace/artifacts 与 benchmark audit metrics；默认关闭，warning-only 不改变 route/pass-fail，strict 仅在 evaluation expectation mismatch 时失败。新增 semantic regression fixture 与 8 个测试文件，覆盖 unknown code normalize、fallback、secret/map compact、warning-only/strict 评估。验证：targeted semantic tests `11 passed`；semantic fake benchmark `13/13 pass_rate=100%`。
- **LLM axial loading 坐标与 plug 剪枝修复**：修复真实 unseen-model run 中 LLM 已生成 `loading_3B` 时 assembler 提前返回，导致 1-based overrides 被 renderer 当成 0-based、且 `thimble_plug` 错误覆盖 active fuel 的问题。已有 loading 现在会按 pin_map 坐标约定归一化，plug-like 非 absorber/control/poison 插入件会从 active-fuel loading 剪枝，base guide tube 保水。验证：real failed patches 复验 `loading_3B` 仅 16 个 0-based Pyrex、`(3,9)/(6,6)/(9,3)` 为 guide_tube；renderer active lattice 16 Pyrex / 0 thimble；`611 passed, 1 skipped, 347 deselected` + compileall + fake workflow 11/11 + regression diff 无回归。
- **空 base loading 覆盖修复**：修复未知模型归一化后 active fuel 仍引用 LLM 生成的空 `base_loading`，导致 Pyrex loading 未应用、3B 退化为 3A-like 水导向管的问题。assembler 现在允许空 loading 被真实 insert loading 替换；真实旧 patches 复验 active fuel `loading_id=pyrex_rod_loading`，renderer 派生 lattice 同时含 `pyrex_rod` 与 `guide_tube`。Targeted tests `16 passed`。
- **未知模型 guide-tube base normalization**：修复真实 incremental/LLM 路径仍会把有限轴向插入件写进 base pin_map 的问题；assembler 现在通用地将 `pyrex_rod_coords`/`thimble_plug_coords` 归一化为 guide-tube base lattice，并为 Pyrex 生成 `lattice_loadings`。真实旧 `data/runs/VERA3_3B` patches 复验：`(3,9)/(6,6)/(9,3)` 组装为 `guide_tube`；targeted tests `55 passed`。
- **VERA3 3B guide-tube water preservation**：修复 3B reference/fixture 将 Pyrex/套管塞坐标直接写入全高 base pin map 的问题；base lattice 改为 24 个水填充 guide_tube，Pyrex 通过 `axial_layers.lattice_loadings` 仅在毒物棒轴向段覆盖，`(3,9)/(6,6)/(9,3)` 等套管塞坐标在 active fuel 中段保持水。新增 axial layer patch 对 `lattice_loadings`/`loading_id` 的通用装配支持。Targeted tests `47 passed`（3B incremental + patch generator）。

### 2026-07-09

- **P0-H Evaluation schema audit + report diff gate**：审计确认 EvaluationCase / EvaluationMetrics / evaluate_trace_against_case / aggregate_evaluation_results / workflow_benchmark report writer 之间字段一致（所有 P0 字段已存在：forbidden_issue_codes、expected_planning_mode、expected_incremental_patch_types、expected_artifact_keys、4 个 rate 指标）。新增 `scripts/diff_evaluation_reports.py`：比较两个 evaluation_report.json，输出 markdown diff（metric delta / case status change / new failure / fixed case），支持 `--fail-on-regression` PR gate（pass_rate / plan_schema / artifact_completeness 下降或新增 failed case → exit 1）。Makefile 新增 `diff-workflow-reports` / `gate-workflow-regression`。新增 9 个 diff 测试。no-OpenMC 全量 `609 passed, 1 skipped`。
- **Requirement resolution bug fix**：requirement 只含文件路径时 feature detection 看不到内容 → pin_map/axial_overlays 任务缺失 → assembly.missing_patch。新增 `requirement_resolver.py`（inline 本地 .md/.txt/.json 内容，不读 URL）；graph `_receive_requirement` 自动解析；executor `required_patch_types_for_state` 增加 benchmark variant 信号。新增 14 个测试。VERA3 3A 真实 LLM 验证：全部 7 patch 生成成功。
- **P0-NEW-1 受控合金 composition library**：新增 `material_library.py`（Zircaloy-4/SS304/Inconel-718 nominal 成分，sum=1.0，alias resolver）+ `material_policy.py`（preserve_plan/apply_alloy_library/strict_confirmed_only 三策略，默认 apply_alloy_library）；assembler 接入 policy 并产出 `material_composition_report.json`；graph/executor 透传 material_policy；新增 `scripts/compare_material_policies.py` (dry-run/OpenMC 两模式)。VERA3 3A/3B fixtures 的三合金 `composition_status` 改为 `needs_library`。新增 70 个测试覆盖 library/policy/CLI；no-OpenMC 全量 `586 passed, 1 skipped`；OpenMC-marked `71 passed`。安全边界：composition 来自公开 handbook midpoint，非 benchmark-specific。
- **Environment validation / OpenMC gating**：新增环境自检脚本、Makefile 测试分层、pytest markers 和 OpenMC importorskip gate；base Python 缺失 OpenMC 时 no-OpenMC/full 测试不再 collection error。`508 passed, 28 skipped`（base full test-all）。
- **P0 Evaluation Backbone**：扩展 EvaluationCase contract、fixture P0 cases、trace evaluator/aggregate metrics，并新增 plan-only `workflow_case_runner` + benchmark-compatible adapter；默认不跑 OpenMC。`compileall` + OpenMC-stub targeted tests。
- **云端环境配置**：新增 `environment.yml`、`Dockerfile`、`.devcontainer/devcontainer.json` 与 `docs/cloud_environment.md`，支持 Conda/Mamba、Docker、Dev Container/Codespaces 三种云端启动方式；README 增加云端环境入口。`git diff --check` + targeted pytest smoke。
- **VERA3 3A/3B 端到端成功**：两个变体均通过 incremental plan builder（真实 LLM deepseek:deepseek-chat）端到端成功运行。全部 7 个 patch 由 LLM 直接生成（不依赖 reference fixture）。3B keff=0.979，3A keff≈1.149。关键修复：assembler 自动构建 ZCylinder surfaces/regions（解决 region=None 几何缺失）；边界条件从实际几何推导（radial=reflective）；元素符号自动路由 add_element；coord_overlap 确定性修复；count_mismatch 对 overlap-repaired 组跳过；零丰度核素跳过；plot 范围从实际几何推导。`841 passed, 3 skipped`。

### 2026-07-08

- **Few-shot 双轨增强**：4 个 anonymized gold case（`data/few_shot_cases/`）接入 monolithic（slim IR+digest）+ incremental（patch 参考段，2400 字符/层预算）路径；`few_shots.py` 按结构特征堆型无关打分合并 gold case；修 `_compact_context` 静默丢弃 `requirement` 的 P0 bug。新增 `few_shot_cases.py` + `scripts/build_few_shot_case.py`。`818 passed`。
- **复杂模型默认按未见处理**：graph 默认 `reference_patch_policy="off"`，VERA3 3B 不再默认走 reference fixture；`prefer_reference_for_structural` 改为 reference 缺失/失败时继续 input-driven synthesis；pin_map count validation 忽略非 pin-map 计数字段（如 spacer grid count）。`48 + 39 targeted passed`，`compileall openmc_agent` 通过。
- **VERA3 3B reference 路由加固**：reference matcher 增加通用 benchmark id normalization（如 `VERA_PROBLEM_3 -> VERA3`）；显式 `reference_only_for_structural` 缺失或验证失败不再落回 LLM；incremental artifacts 写入前清理 stale patch/result 文件。`841 passed, 3 skipped`。
- **VERA3 3B incremental/reference 修复**：区分 partial/complete `expected_counts`，新增 deterministic actual pin counts；`grid_zircaloy4→zircaloy4` alias resolver + overlay canonicalization；VERA3 3B reference structural path 保留为显式策略并禁止 monolithic reflect fallback。`831 passed, 3 skipped`。
- **VERA3 3B expected_counts 二次加固**：assembler 最终写入前 reconcile FactsPatch 计数与 deterministic expanded pin_map；3B pattern 自洽时以 actual counts 覆盖错误 `guide_tube=24` facts，避免 `expected_counts sum 313` 阻断渲染。`832 passed, 3 skipped`。
- **VERA3 3B axial-layer material alias 加固**：material resolver 支持唯一 variant suffix 解析（如 `borated_water -> borated_water_3B`）；assembler canonicalize axial layer fill material，避免 reference structural axial layers 与 LLM material ids 命名不一致导致 skeleton/reflect。`833 passed, 3 skipped`。
- **Incremental Plan Builder Phase 0–7D**：复杂 3D assembly 从 monolithic 25K JSON 改为 7-patch 分层生成 + 确定性组装（`plan_builder/`）；graph dispatcher 路由（`use_incremental_executor=True` 默认）；reference-backed deterministic structural patches + resumable execution；真实 LLM 加固（禁不安全 monolithic fallback、strict patch-output contract、`json_schema` structured mode、attempt artifact 可见）。测试 `453→791 passed`。

### 2026-07-07

- **3D assembly guard + Level 1 overlay（Step 1–6）**：guard 阻断 3D 压扁为 2D（四个 `assembly3d.*` issue code）；`AxialOverlaySpec` IR；Level 1 `homogenized_open_region` renderer（保 pin/tube through-path）；skeleton overlay 自动提升；VERA3 acceptance foundation（reference fixture + benchmark validator + 三层 E2E 测试）。`490→581 passed`。
- **source/bounds 一致性（Step 5–6）**：source z 绑定活性燃料、xy 绑定 full footprint；plot origin 重定心修 quarter-plot bug；新增 `source_settings.py` / `geometry_bounds.py` + 17 个 runtime/bounds issue code。`549→565 passed`。
- **overlay cell 复用 bug 修复**：overlay universe 克隆 solid cell 而非复用，修 base 燃料段被抢走导致的 source rejection（真实根因）。`567 passed`。
- **专家验证工具**：`verification.py` digest（不变量检查表 + pin/axial/overlay/material/bounds）+ 3D voxel plot（ParaView 可读）。`581 passed`。
- **runtime/渲染修复**：surface 轴截距别名（`x→x0`/`y→y0`/`z→z0`）、`rectangular_prism` pitch kwarg、dangling lattice outer 降级、suppress 已配置的 `OPENMC_CROSS_SECTIONS` capability 阶段专家问题。`533→574 passed`。
- **Knowledge Asset Runtime Loader**：orchestrator GraphRAG stage 加载持久化 knowledge graph（path/env fallback，失败只 warning）；trace summary 暴露 node/edge/source/warning 计数。`474 passed`。
- **Knowledge Ingestion + GraphRAG reranker/query planner**：ingestion pipeline、evidence ranker + dedup + prompt budget、query planner + graph path scoring、docs 整理、repo-local `AGENTS.md`/`CLAUDE.md` 自动提交规则。`453 passed`。
- **case3 输入修正**：导向管/裂变室填水（C5G7 未插棒构型，避免高估 Zr 吸收）。

> 更早期的检索/retrieval/graphrag 基础设施搭建记录见 git history；本报告聚焦 2026-07-07 起的工程进展。

### P0-D/E/F Workflow Benchmark Reporting

The P0 evaluation backbone now includes a report-generating workflow benchmark entry point. `openmc_agent.workflow_benchmark.run_workflow_benchmark(...)` loads the evaluation case manifest, runs each case through the fake or opt-in workflow runner, evaluates traces, aggregates metrics, and writes `evaluation_report.json`, `benchmark_summary.md`, per-case traces, and per-case artifact summaries.

The CLI `scripts/run_workflow_benchmark.py` defaults to `model=fake`, `mode=plan-only`, rendering disabled, and OpenMC tools disabled. Non-fake models are refused unless `--allow-real-llm` is passed, keeping the no-OpenMC/no-LLM test boundary intact. The primary local command is `make benchmark-fake`; `make benchmark-check` compares the fake result with the tracked fixture baseline at `tests/fixtures/workflow_baseline/evaluation_report.json`.

- 2026-07-09 — P0-D/E/F workflow benchmark reporting: added workflow benchmark API, CLI report generation, fake benchmark Make target, and no-OpenMC tests for reports/traces. Validation: targeted workflow benchmark tests plus no-OpenMC suite.
## 2026-07-17 — Phase 1C canonical planning safety boundary

- Added feature-to-Facts reconciliation, one persisted canonical planning scope,
  and canonical task-plan persistence so task selection and assembly share the
  same patch-family decision.
- Added source-critical localized-insert/profile checks and grouped material
  density execution-readiness diagnostics; focused regression tests pass.
- Boundaries remain unchanged: no generic retry controller and no additional
  Placement, Material–Universe, Axial, or Final Plan LLM gate is claimed.
## 2026-07-17 — Phase 3 executable dependency retry foundation

Added protocol v0.5 typed retry requests, owner routing, a single patch
dependency graph, clone/atomic owner transaction, and resumable invalidation
ledger.  Added reactor-neutral protocol and graph tests; no new reviewer,
runtime repair, or monolithic fallback path was introduced.

## 2026-07-17 — Real-model JSON output and targeted patch recovery

- Added opt-in CLI/client controls for JSON-object/schema mode, per-call output
  tokens, and provider reasoning effort; actual structured fallback mode is
  retained in patch/review artifacts. Defaults remain provider-compatible.
- Patch-generation failure now preserves the source requirement and valid
  envelopes, resuming at the failed patch instead of contaminating every prompt
  with a full-plan correction. Hyphenated document identifiers no longer count
  as lattice dimensions. Focused regression tests: 65 passed.

## 2026-07-18 — Placement Gate deferred applicability

- Placement Gate now remains pending while candidate profile/intent patches are
  still being generated, and only becomes `skipped` after task-plan completion.
  A stale `not_applicable` checkpoint is explicitly reopened if its inputs later
  become applicable; this prevents an illegal `skipped → reviewing` transition.
- Validation: placement, incremental, graph, and controller regressions: 41 passed.

## 2026-07-18 — Axial replacement-universe dependency replay

- An axial layer that references a missing replacement universe now invalidates
  the `universes` owner, its dependency-graph descendants, and the failed
  axial consumer before Graph replay. The next pass regenerates that bounded
  path instead of skipping a stale valid `UniversesPatch` three times.
- Validation: incremental executor, retry router, graph replay, and repair
  regressions: 49 passed.

## 2026-07-18 — Axial patch admission and assembly diagnostics

- Incomplete axial intervals and unattached lattice loadings now fail patch
  validation before assembly; executor results retain the assembler's concrete
  issue codes alongside the generic failure wrapper.
- Validation: focused patch-validator and incremental-executor regressions.

## 2026-07-18 — Controlled Facts resume barrier and evidence coverage

- Coalesced line-aligned Markdown evidence paragraphs before applying the Facts
  review chunk limit; a controlled gate failure is terminal for ordinary Graph
  regeneration and cannot be bypassed by reusing a valid FactsPatch.
- Placement dependency requests now retain required universe IDs and gate-input
  hash; focused tests plus the non-OpenMC/non-real-LLM suite and fake workflow
  benchmark passed.

## 2026-07-18 — Axial JSON schema recovery and bounded controlled retry

- Axial prompts now enumerate canonical layer roles and forbid null ordering
  priorities; two exact lexical role aliases and an omitted-default priority
  are normalized with attempt-level audit records only.
- Controlled Graph execution stops after the failed patch's local budget is
  exhausted, retains its checkpoint/candidate hashes, and does not reopen a
  whole-plan retry. Validation: focused generator, executor, and Graph tests.

## 2026-07-18 — Executable closed-loop CLI defaults

- The CLI now defaults to controlled incremental planning with all currently
  executable gates enabled: Facts, Material–Universe, and Placement. Per-gate
  modes inherit the selected loop mode; patch output remains provider `auto`.
- Axial and Assembled Plan gate IDs remain opt-in/unimplemented rather than
  causing a default modeling command to fail. Validation: CLI default tests.

## 2026-07-20 — Phase 8C Step 2D structured-output robustness closure

- Added a shared two-attempt structured-output transaction with immutable payload
  hashes, attempt telemetry, stale-output detection, and budget accounting;
  all five gate reviewers and investigation planning use the same seam.
- Investigation results now retain planner retry/hash telemetry and input-driven
  Facts/Materials/Universes semantic coverage; redundant actions stop after
  coverage completion while uncovered targets remain fail-closed.
- Added resume-policy hashing, truthfulness checks, README/design documentation,
  and regression tests. OpenMC 0.15.3 is now available under conda run through
  the persisted ENDF/B-VII.1 cross-section configuration.
- Validation: full non-OpenMC/non-LLM suite 3365 passed, 2 skipped, 392
  deselected; compileall passed; fake workflow 21/21; offline VERA4 mutations
  6/6. No baseline report is present.
- Canary boundary: all three authorized 40-minute VERA4 attempts were rejected before
  execution by the tenant external-data transfer policy. Step 2D is not declared
  complete and no real-model acceptance claim is made.

## 2026-07-21 — Phase 8C Step 2E Facts closure and baseline early-stop

- Facts revision now performs up to three clone/rereview closure rounds through the shared structured-output transaction; only a clean rereview commits, while duplicate, incomplete, and human-only closures fail with stable codes.
- Completeness review keeps downstream material/geometry/universe targets nonblocking at Facts, and mandatory investigation baseline coverage now skips the planner entirely.
- Validation: focused Facts/investigation regressions 57 passed; full non-OpenMC/non-LLM suite and compileall passed; fake workflow 21/21. Baseline diff skipped because the baseline report is absent; OpenMC 0.15.3 imports in `openmc-env`.

## 2026-07-21 - Facts metadata finding classification normalization

- Facts review normalization now classifies source-backed findings confined to
  missing_facts, assumptions, or source_notes as append-only metadata repairs,
  with auditable original reviewer flags. This records uncertainty; it never
  authorizes inferred physical values or changes mixed-path findings.
- The executor no longer carries a divergent rereview-only override. Validation:
  focused Facts regressions 21 passed; full non-OpenMC/non-LLM suite and
  compileall passed; fake workflow benchmark 21/21. Baseline diff is unavailable
  because no baseline report is present.
