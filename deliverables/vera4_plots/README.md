# VERA4 渲染图交付

来源运行：

`data/runs/phase8c_step3n_vera4_redraw_v1/workflow`

该运行使用 deterministic VERA4 source-backed fixture 离线组装，不调用 LLM。渲染状态为 `renderability=runnable`，XML export 与 `openmc -p` 均成功。新 plan 显式使用径向 reflective、轴向 vacuum 六面边界，包含 fuel plenum/endplug/water_pin、RCCA localized insert 与实体 spacer-grid materialization。原始 `model.py`、XML、plots 和 `simulation_plan.json` 保留在 run 目录。

## 文件说明

- `vera4_render_contact_sheet.png`：关键渲染切片的总览拼图，便于快速审阅。
- `full_core_xy_active_fuel_material.png`：活性燃料中平面 XY 切片，按 material 着色。
- `full_core_xy_active_fuel_cell.png`：活性燃料中平面 XY 切片，按 cell 着色。
- `full_core_xy_localized_insert_material.png` / `full_core_xy_localized_insert_cell.png`：穿过中心 RCCA 控制棒插入段的 XY 切片。
- `center_assembly_xy_localized_insert_material.png` / `center_assembly_xy_localized_insert_cell.png`：中心组件控制棒插入段局部放大。
- `full_core_xy_grid_mid_material.png`：grid midplane XY 切片，按 material 着色。
- `full_core_xy_grid_mid_cell.png`：grid midplane XY 切片，按 cell 着色。
- `center_assembly_xy_grid_mid_material.png` / `center_assembly_xy_grid_mid_cell.png`：中心组件 spacer-grid 平面局部放大。
- `full_core_xz_material.png`：全堆 XZ 轴向切片，按 material 着色。
- `full_core_xz_cell.png`：全堆 XZ 轴向切片，按 cell 着色。
- `center_assembly_xz_full_height_material.png` / `center_assembly_xz_full_height_cell.png`：中心组件全高 XZ 放大图，可检查燃料棒上部气腔、端塞、RCCA 轴向分段。
- `center_assembly_xz_lower_interface_*` / `center_assembly_xz_upper_interface_*`：组件与下/上结构界面 XZ 放大图。
- `verification_xz_material.png`：高纵横比 verification XZ 切片，按 material 着色。
- `verification_xz_cell.png`：高纵横比 verification XZ 切片，按 cell 着色。

说明：`render_detail_manifest.json` 记录完整 plot 文件列表、source run 和 plan summary。
