# OpenMC-Agent 基准题演示结果清单

由 `scripts/collect_demo_results.py` 自动生成。keff 为**中等统计量诊断值**，
用于验证模型可运行，**非**基准标准值（C5G7 连续能组成亦非七群参考数据）。

| 算例 | 模式 | renderability | renderer | keff ± σ | 状态 | plots |
|---|---|---|---|---|---|---|
| C5G7 | monolithic（LLM 单次 plan，Gate 关） | runnable | core | 1.22101 ± 0.00330 | runnable（keff=1.22101±0.00330） | 6 |
| VERA2_2A | 增量 + Gate 关 | none | — | — | 未完成：incremental.patch_generation_failed | 0 |
| VERA2_2A_mono | monolithic（LLM 单次 plan，Gate 关） | skeleton | — | — | skeleton（不可导出）：assembly3d.spacer_grid_overlay_required | 0 |
| VERA3_3B | 增量 + Gate 关 | none | — | — | 未完成：lattice.pin_count_mismatch | 0 |
| VERA3_3B_gate | 增量 + Gate 开（controlled 探针） | none | — | — | 未完成：planning.material_universe_gate_not_accepted | 0 |
| VERA3_3B_mono | monolithic（LLM 单次 plan，Gate 关） | skeleton | — | — | skeleton（不可导出）：assembly3d.axial_layers_required；assembly3d.default_z_extent_for_axial_problem | 0 |
| VERA3_3B_reference | 参照（proven build + fresh 中等统计量 transport） | runnable | assembly | 0.96762 ± 0.00193 | runnable（keff=0.96762±0.00193） | 0 |

- 阻塞码含义：`fullcore.fuel_variant_unreachable` = 增量装配阶段燃料变体不可达；`planning.material_universe_gate_not_accepted` = Material–Universe Gate 未通过；`assembly3d.spacer_grid_overlay_required` / `axial_layers_required` = 3D 轴向/格架 guard 降级为 skeleton。

## 各算例产物路径

- **C5G7** (monolithic（LLM 单次 plan，Gate 关）): `data/runs/demo/C5G7/`
- **VERA2_2A** (增量 + Gate 关): `data/runs/demo/VERA2_2A/`
- **VERA2_2A_mono** (monolithic（LLM 单次 plan，Gate 关）): `data/runs/demo/VERA2_2A_mono/`
- **VERA3_3B** (增量 + Gate 关): `data/runs/demo/VERA3_3B/`
- **VERA3_3B_gate** (增量 + Gate 开（controlled 探针）): `data/runs/demo/VERA3_3B_gate/`
- **VERA3_3B_mono** (monolithic（LLM 单次 plan，Gate 关）): `data/runs/demo/VERA3_3B_mono/`
- **VERA3_3B_reference** (参照（proven build + fresh 中等统计量 transport）): `data/runs/demo/VERA3_3B_reference/`
