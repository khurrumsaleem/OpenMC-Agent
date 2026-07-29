#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# OpenMC-Agent 基准题演示驱动（VERA2 2A / VERA3 3B / C5G7）
# -----------------------------------------------------------------------------
# 本脚本是"可执行的操作指南"：透明地调用 run_model.py / inspect / openmc，
# 把三个基准题跑到「可运行模型 + 中等统计量 keff」，产物写入 data/runs/demo/。
#
# 策略（与 README "基准题演示" 章节一致）：
#   - C5G7        : monolithic —— LLM 单次输出整个 plan（不走增量、不开 Gate）
#   - VERA2/VERA3 : 增量建模 —— 主展示 run 用 Gate 关（run_model.py），
#                   另用 inspect --plan-loop-mode controlled 做 Gate 可行性探针
#                   （开启 Gate 可能被阻塞，阻塞即如实记录，不强求出 transport）
#
# 保真度：中等统计量。run_model.py/inspect 的 --smoke-test 先验证几何（100 粒子），
# 随后 transport 步把 settings.xml 调到 ~万级粒子再 openmc，得到可信 keff。
#
# 用法（须在 openmc-env 中运行）：
#   conda run --no-capture-output -n openmc-env bash scripts/run_demo.sh [target]
#     target = all (默认) | c5g7 | vera3 | vera2 | vera3-gate | vera2-gate
#
# 可通过环境变量覆盖：
#   MODEL=deepseek:deepseek-chat  LLM 模型（provider:model）
#   PARTICLES / BATCHES / INACTIVE 中等统计量输运参数（全局默认）
# -----------------------------------------------------------------------------
set -u

MODEL="${MODEL:-deepseek:deepseek-chat}"
PARTICLES="${PARTICLES:-10000}"
BATCHES="${BATCHES:-40}"
INACTIVE="${INACTIVE:-20}"
# C5G7 四分之一堆芯几何更大，单独降一些粒子以控制时间
C5G7_PARTICLES="${C5G7_PARTICLES:-5000}"
C5G7_BATCHES="${C5G7_BATCHES:-30}"
C5G7_INACTIVE="${C5G7_INACTIVE:-15}"

DEMO_ROOT="data/runs/demo"
PY="${PYTHON:-python}"
TARGET="${1:-all}"

log() { printf '\n\033[1;36m[run_demo]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[run_demo WARN]\033[0m %s\n' "$*" >&2; }

# --- 前置检查：必须在 openmc-env 里跑（有 openmc）---
if ! "$PY" -c 'import openmc' >/dev/null 2>&1; then
  warn "当前 python 无法 import openmc。请用: conda run --no-capture-output -n openmc-env bash scripts/run_demo.sh"
fi
mkdir -p "$DEMO_ROOT"

# -----------------------------------------------------------------------------
# 单算例建模函数
# -----------------------------------------------------------------------------
# Gate 关 + 增量（run_model.py 默认），带 smoke 验证几何
gates_off_run() {
  local input="$1" benchmark="$2" variant="$3" outd="$4"
  log "VERA $benchmark $variant — 增量建模，Gate 关 (run_model.py) → $outd"
  "$PY" scripts/run_model.py \
    --input "$input" --benchmark "$benchmark" --variant "$variant" \
    --model "$MODEL" --allow-real-llm --smoke-test \
    --out "$outd" || warn "$benchmark $variant gates-off 建模失败（继续）"
}

# monolithic：LLM 单次出整个 plan（C5G7 / VERA），Gate 关
# 参数: <input> <benchmark> <variant|-> <outd>
monolithic_run() {
  local input="$1" benchmark="$2" variant="$3" outd="$4"
  log "$benchmark ${variant#-} — monolithic 单次 plan，Gate 关 (run_model.py --no-incremental) → $outd"
  local -a args=(--input "$input" --benchmark "$benchmark" \
    --model "$MODEL" --allow-real-llm --no-incremental --smoke-test --out "$outd")
  [[ "$variant" != "-" && -n "$variant" ]] && args+=(--variant "$variant")
  "$PY" scripts/run_model.py "${args[@]}" || warn "$benchmark ${variant#-} monolithic 建模失败（继续）"
}

# Gate 开探针（inspect controlled）——验证 Gate 路径是否阻塞
gate_probe_run() {
  local input="$1" state="$2" outd="$3"
  log "VERA state $state — Gate 开探针 (inspect --plan-loop-mode controlled) → $outd"
  "$PY" -u -m openmc_agent.inspect \
    --plan --verbose --md-file "$input" --state "$state" \
    --model "$MODEL" --plan-loop-mode controlled \
    --smoke-test --output-dir "$outd" \
    || warn "state $state Gate 探针未走通（可能即 Gate 阻塞，属预期可观察项）"
}

# -----------------------------------------------------------------------------
# 中等统计量输运：patch settings.xml → openmc → 打印 keff
# 参数: <run_dir> <particles> <batches> <inactive>
# -----------------------------------------------------------------------------
transport() {
  local dir="$1" p="$2" b="$3" i="$4"
  local settings="$dir/settings.xml"
  if [[ ! -f "$settings" ]]; then
    warn "跳过 transport：$settings 不存在（建模未到导出阶段）"
    return
  fi
  log "中等统计量输运 $dir — particles=$p batches=$b inactive=$i"
  "$PY" - "$dir" "$p" "$b" "$i" <<'PY' || warn "transport 失败（继续）"
import sys, os, glob
import xml.etree.ElementTree as ET
d, p, b, i = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
path = os.path.join(d, "settings.xml")
tree = ET.parse(path); root = tree.getroot()
def settext(tag, val):
    el = root.find(tag)
    if el is None:
        el = ET.SubElement(root, tag)
    el.text = str(val)
settext("particles", p); settext("batches", b); settext("inactive", i)
tree.write(path)
print(f"[transport] patched {path}: particles={p} batches={b} inactive={i}")
import openmc
openmc.run(cwd=d, output=True)
sps = sorted(glob.glob(os.path.join(d, "statepoint.*.h5")))
if sps:
    import h5py
    with h5py.File(sps[-1], "r") as f:
        kc = f["k_combined"][()]
        print(f"[transport] KEFF {kc[0]:.5f} +/- {kc[1]:.5f}  ({os.path.basename(sps[-1])})")
else:
    print("[transport] 无 statepoint 产出")
PY
}

# -----------------------------------------------------------------------------
# 各算例
# 注：当前增量装配存在 fuel_variant_unreachable 的已知问题（Gate 开/关的增量
# 路径都会命中），故 VERA 同时提供增量(vera2/vera3)与 monolithic(*-mono)两条路径。
# monolithic 走单次 plan，绕开增量装配，是目前 fresh 建模最可能跑通的方式。
# -----------------------------------------------------------------------------
run_c5g7() {
  monolithic_run "Input/case3.md" "C5G7" "-" "$DEMO_ROOT/C5G7"
  transport "$DEMO_ROOT/C5G7" "$C5G7_PARTICLES" "$C5G7_BATCHES" "$C5G7_INACTIVE"
}

run_vera3() {  # 增量 + Gate 关（可能命中装配 bug）
  gates_off_run "Input/VERA3_problem.md" "VERA3" "3B" "$DEMO_ROOT/VERA3_3B"
  transport "$DEMO_ROOT/VERA3_3B" "$PARTICLES" "$BATCHES" "$INACTIVE"
}
run_vera3_mono() {  # monolithic + Gate 关（推荐 fresh 路径）
  monolithic_run "Input/VERA3_problem.md" "VERA3" "3B" "$DEMO_ROOT/VERA3_3B_mono"
  transport "$DEMO_ROOT/VERA3_3B_mono" "$PARTICLES" "$BATCHES" "$INACTIVE"
}

run_vera3_gate() {
  gate_probe_run "Input/VERA3_problem.md" "3B" "$DEMO_ROOT/VERA3_3B_gate"
}

run_vera2() {  # 增量 + Gate 关（可能命中装配 bug / universes pyrex）
  gates_off_run "Input/VERA2_problem.md" "VERA2" "2A" "$DEMO_ROOT/VERA2_2A"
  transport "$DEMO_ROOT/VERA2_2A" "$PARTICLES" "$BATCHES" "$INACTIVE"
}
run_vera2_mono() {  # monolithic + Gate 关（推荐 fresh 路径）
  monolithic_run "Input/VERA2_problem.md" "VERA2" "2A" "$DEMO_ROOT/VERA2_2A_mono"
  transport "$DEMO_ROOT/VERA2_2A_mono" "$PARTICLES" "$BATCHES" "$INACTIVE"
}

run_vera2_gate() {
  gate_probe_run "Input/VERA2_problem.md" "2A" "$DEMO_ROOT/VERA2_2A_gate"
}

case "$TARGET" in
  all)
    # C5G7 + VERA monolithic（最可能跑通）+ 增量 & Gate 探针（记录当前边界）
    run_c5g7
    run_vera3_mono; run_vera3; run_vera3_gate
    run_vera2_mono; run_vera2; run_vera2_gate
    ;;
  c5g7)        run_c5g7 ;;
  vera2)       run_vera2 ;;
  vera2-mono)  run_vera2_mono ;;
  vera2-gate)  run_vera2_gate ;;
  vera3)       run_vera3 ;;
  vera3-mono)  run_vera3_mono ;;
  vera3-gate)  run_vera3_gate ;;
  *) echo "未知 target: $TARGET"
     echo "可用: all | c5g7 | vera2 | vera2-mono | vera2-gate | vera3 | vera3-mono | vera3-gate"
     exit 2 ;;
esac

log "完成。运行 python scripts/collect_demo_results.py 生成结果清单。"
