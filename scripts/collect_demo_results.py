#!/usr/bin/env python3
"""Collect OpenMC-Agent demo benchmark results into a manifest.

Scans ``data/runs/demo/*/`` and, for each run directory, extracts:
  - keff ± σ from the highest-batch ``statepoint.*.h5`` (h5py, ``k_combined``)
  - renderability / supported_renderer from ``capability_report.json``
  - workflow status from ``workflow_outcome.json``
  - whether model.py / XML / plots were produced

Writes a structured ``data/runs/demo/results.json`` and a human-readable
``data/runs/demo/README.md`` table. Honest about missing/blocked runs — never
fabricates a keff.

Usage:
    conda run -n openmc-env python scripts/collect_demo_results.py [--root data/runs/demo]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_keff(statepoint: Path) -> tuple[float, float] | None:
    try:
        import h5py
    except Exception:
        return None
    try:
        with h5py.File(statepoint, "r") as f:
            kc = f["k_combined"][()]
            return float(kc[0]), float(kc[1])
    except Exception:
        return None


def _latest_statepoint(d: Path) -> Path | None:
    sps = sorted(d.glob("statepoint.*.h5"), key=lambda p: int(p.stem.split(".")[1]))
    return sps[-1] if sps else None


def _load_json(d: Path, name: str) -> dict | None:
    p = d / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _mode_from_dir(name: str) -> str:
    low = name.lower()
    if low.endswith("_reference"):
        return "参照（proven build + fresh 中等统计量 transport）"
    if low.endswith("_mono"):
        return "monolithic（LLM 单次 plan，Gate 关）"
    if low.endswith("_gate"):
        return "增量 + Gate 开（controlled 探针）"
    if "c5g7" in low:
        return "monolithic（LLM 单次 plan，Gate 关）"
    return "增量 + Gate 关"


def _classify(rec: dict) -> str:
    if rec["keff"] is not None:
        return f"runnable（keff={rec['keff'][0]:.5f}±{rec['keff'][1]:.5f}）"
    blk = "；".join(rec["blocker"]) if rec["blocker"] else ""
    if rec["renderability"] == "skeleton":
        return f"skeleton（不可导出）{('：'+blk) if blk else ''}"
    if rec["xml_ok"]:
        return "exportable（已导出 XML，无 keff）"
    if blk:
        return f"未完成：{blk}"
    if rec["model_exists"]:
        return "skeleton（仅 model.py，未导出）"
    return "未完成（无 model 产物）"


def collect(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        return rows
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        cap = _load_json(d, "capability_report.json") or {}
        outcome = _load_json(d, "workflow_outcome.json") or {}
        inc = _load_json(d, "incremental/incremental_execution_result.json") or {}
        sp = _latest_statepoint(d)
        keff = _read_keff(sp) if sp else None
        plots = list((d / "plots").glob("*.png")) if (d / "plots").is_dir() else []
        # blocker reason codes: prefer workflow reason_codes, then incremental errors
        blocker: list[str] = list(outcome.get("reason_codes") or [])
        if not blocker:
            blocker = [i.get("code") for i in (inc.get("issues") or [])
                       if i.get("severity") == "error" and i.get("code")]
        rec = {
            "case": d.name,
            "mode": _mode_from_dir(d.name),
            "renderability": cap.get("renderability") or outcome.get("renderability"),
            "supported_renderer": cap.get("supported_renderer"),
            "workflow_status": outcome.get("status"),
            "blocker": blocker,
            "model_exists": (d / "model.py").exists(),
            "xml_ok": all((d / x).exists() for x in ("materials.xml", "geometry.xml", "settings.xml")),
            "statepoint": sp.name if sp else None,
            "keff": keff,
            "plots_count": len(plots),
            "dir": str(d.relative_to(REPO_ROOT)) if REPO_ROOT in d.parents else str(d),
        }
        rec["outcome"] = _classify(rec)
        rows.append(rec)
    return rows


def write_manifest(root: Path, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# OpenMC-Agent 基准题演示结果清单",
        "",
        "由 `scripts/collect_demo_results.py` 自动生成。keff 为**中等统计量诊断值**，",
        "用于验证模型可运行，**非**基准标准值（C5G7 连续能组成亦非七群参考数据）。",
        "",
        "| 算例 | 模式 | renderability | renderer | keff ± σ | 状态 | plots |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        keff = f"{r['keff'][0]:.5f} ± {r['keff'][1]:.5f}" if r["keff"] else "—"
        lines.append(
            f"| {r['case']} | {r['mode']} | {r['renderability'] or '—'} | "
            f"{r['supported_renderer'] or '—'} | {keff} | {r['outcome']} | "
            f"{r['plots_count']} |"
        )
    lines.append("")
    lines.append("- 阻塞码含义：`fullcore.fuel_variant_unreachable` = 增量装配阶段燃料变体不可达；"
                 "`planning.material_universe_gate_not_accepted` = Material–Universe Gate 未通过；"
                 "`assembly3d.spacer_grid_overlay_required` / `axial_layers_required` = 3D 轴向/格架 guard 降级为 skeleton。")
    lines += [
        "",
        "## 各算例产物路径",
        "",
    ]
    for r in rows:
        lines.append(f"- **{r['case']}** ({r['mode']}): `{r['dir']}/`")
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/runs/demo", help="demo runs root dir")
    args = ap.parse_args(argv)
    root = Path(args.root)
    rows = collect(root)
    if not rows:
        print(f"未在 {root} 找到任何算例目录。请先运行 scripts/run_demo.sh。")
        return 1
    write_manifest(root, rows)
    print(f"已生成 {root / 'results.json'} 与 {root / 'README.md'}（{len(rows)} 个算例）：")
    for r in rows:
        keff = f"keff={r['keff'][0]:.5f}" if r["keff"] else "no-keff"
        print(f"  - {r['case']:<18} {r['renderability'] or '—':<12} {keff:<20} {r['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
