# VERA4 渲染图交付

来源运行：

`data/runs/phase8c_step3m_vera4_openmc_smoke_seed_v2`

该运行状态为 `CAMPAIGN_PASSED`，单 run 为 `FIRST_PASS_SUCCESS`。五个 planning gates 均 accepted，`renderability=runnable`，`xml_exported=true`，`geometry_debug_passed=true`，`smoke_passed=true`，`llm_call_count=0`。本目录只整理代表性 PNG；原始 OpenMC XML、statepoint、summary 和完整 plot artifacts 保留在 run 目录。

## 文件说明

- `vera4_render_contact_sheet.png`：12 个渲染切片的总览拼图，便于快速审阅。
- `full_core_xy_active_fuel_material.png`：活性燃料中平面 XY 切片，按 material 着色。
- `full_core_xy_active_fuel_cell.png`：活性燃料中平面 XY 切片，按 cell 着色。
- `full_core_xy_grid_mid_material.png`：grid midplane XY 切片，按 material 着色。
- `full_core_xy_grid_mid_cell.png`：grid midplane XY 切片，按 cell 着色。
- `full_core_xz_material.png`：全堆 XZ 轴向切片，按 material 着色。
- `full_core_xz_cell.png`：全堆 XZ 轴向切片，按 cell 着色。
- `verification_xz_material.png`：高纵横比 verification XZ 切片，按 material 着色。
- `verification_xz_cell.png`：高纵横比 verification XZ 切片，按 cell 着色。

说明：lower/upper structural XY 切片在原始 artifacts 中存在，但为单色结构层图，本目录未作为主展示图单列；可在原始 `geometry_debug/plots/` 中查看。
