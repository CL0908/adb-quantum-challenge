// Build-time guard for the Banker view: every number the page reads must exist in dashboard.json / grid.json.
import { readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const read = (f) => {
  try {
    return JSON.parse(readFileSync(join(root, 'public/data', f), 'utf8'))
  } catch (e) {
    console.error(`check-dashboard: cannot read public/data/${f}: ${e.message}\nRun: PYTHONPATH=. python -m src.viz.export_dashboard_data`)
    process.exit(1)
  }
}
const d = read('dashboard.json')
const g = read('grid.json')
const bad = []
const num = (path, x) => { if (typeof x !== 'number' || !Number.isFinite(x)) bad.push(path) }
const str = (path, x) => { if (typeof x !== 'string' || !x.trim()) bad.push(path) }
const need = (path, o, keys, check) => {
  if (!o || typeof o !== 'object') return bad.push(path)
  for (const k of keys) check(`${path}.${k}`, o[k])
}

need('asof', d.asof, ['date', 'commit', 'file'], str)
if (!Array.isArray(d.scenario_order) || d.scenario_order.length !== 5) bad.push('scenario_order (5 scenarios expected)')
const TARGETS = ['expected_loss', 'joint_attachment', 'exceedance', 'stop_loss', 'joint_exhaustion']
for (const id of ['kgtj', 'carec', 'ap']) {
  const p = d.pools?.[id]
  if (!p) { bad.push(`pools.${id}`); continue }
  need(`pools.${id}`, p, ['label', 'short', 'exact_label'], str)
  if (!Array.isArray(p.runs) || !p.runs.length) bad.push(`pools.${id}.runs`)
  for (const [i, r] of (p.runs ?? []).entries()) need(`pools.${id}.runs[${i}]`, r, ['file', 'script', 'use', 'commit', 'date'], str)
  if (id === 'kgtj') {
    for (const s of d.scenario_order ?? []) {
      const sc = p.scenarios?.[s]
      if (!sc) { bad.push(`pools.kgtj.scenarios.${s}`); continue }
      need(`kgtj.${s}`, sc, ['label', 'desc'], str)
      need(`kgtj.${s}`, sc, ['delivered', 'threshold_musd'], num)
      for (const t of TARGETS) need(`kgtj.${s}.${t}`, sc.targets?.[t], ['value', 'brief_rounded', 'rdpo_rel_rmse', 'rdpo_coverage', 'reps', 'budget', 'best_baseline_rel_rmse', 'mc_rel_rmse'], num)
    }
  } else {
    if (!Array.isArray(p.targets) || p.targets.length < 3) bad.push(`pools.${id}.targets`)
    for (const [i, t] of (p.targets ?? []).entries()) {
      num(`pools.${id}.targets[${i}].value`, t.value)
      // a reference without an RDPO run is allowed (shown as "no run"); a partial run is not
      const parts = ['rdpo_rel_rmse', 'rdpo_coverage', 'budget'].map((k) => t[k])
      if (parts.some((x) => x === null) && !parts.every((x) => x === null)) bad.push(`pools.${id}.targets[${i}] partial RDPO run`)
      parts.forEach((x, j) => x !== null && num(`pools.${id}.targets[${i}].rdpo[${j}]`, x))
    }
    if ((p.targets ?? []).filter((t) => t.rdpo_rel_rmse !== null).length < 3) bad.push(`pools.${id}: fewer than 3 targets with an RDPO run`)
    need(`pools.${id}`, p, ['T_moderate', 'T_rare'], num)
    if (id === 'carec' && !(p.curve?.x?.length > 5 && p.curve.x.length === p.curve.y.length)) bad.push('pools.carec.curve')
  }
}
for (const k of ['eq_axis', 'rain_axis']) if (!Array.isArray(g[k]) || !g[k].length) bad.push(`grid.${k}`)
for (const k of ['exceed', 'joint_exhaustion', 'delivered']) {
  const m = g.values?.[k]
  if (!Array.isArray(m) || m.length !== g.eq_axis?.length || m[0]?.length !== g.rain_axis?.length) bad.push(`grid.values.${k} shape`)
}
for (const w of ['brief', 'data']) need(`grid.${w}`, g[w], ['eq', 'rain', 'exceed', 'joint_exhaustion', 'delivered', 'expected_loss', 'shortfall'], num)
for (const r of [0.25, 0.02]) if (!g.rain_axis?.some((x) => Math.abs(x - r) < 1e-9)) bad.push(`grid.rain_axis lacks ${r}`)

if (bad.length) {
  console.error(`\ncheck-dashboard: ${bad.length} missing or invalid field(s):`)
  for (const b of bad) console.error('  ' + b)
  console.error('\nRegenerate with `PYTHONPATH=. python -m src.viz.export_dashboard_data` (src/viz/export_dashboard_data.py).\n')
  process.exit(1)
}
console.log('check-dashboard: dashboard.json and grid.json complete.')
