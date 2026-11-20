// Build-time guard: every value key the source reads must exist in public/data/values.json (and story.json must parse).
// Keys are read as  <V k="KEY" />  and  v.KEY / v['KEY'] (v = useValues()).  Keys read through the name `o` are optional
// (the UI hides that element when absent) and are not required here.
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const walk = (d) => readdirSync(d).flatMap((f) => (statSync(join(d, f)).isDirectory() ? walk(join(d, f)) : [join(d, f)]))
const used = new Map()
for (const f of walk(join(root, 'src')).filter((f) => /\.(tsx?|jsx?)$/.test(f))) {
  const src = readFileSync(f, 'utf8')
  for (const m of src.matchAll(/\bk=["']([A-Z][A-Z0-9_]+)["']|\bv\.([A-Z][A-Z0-9_]+)\b|\bv\[["']([A-Z][A-Z0-9_]+)["']\]/g)) {
    const k = m[1] || m[2] || m[3]
    used.set(k, [...(used.get(k) ?? []), f.replace(root + '/', '')])
  }
}
let values
try {
  values = JSON.parse(readFileSync(join(root, 'public/data/values.json'), 'utf8'))
  JSON.parse(readFileSync(join(root, 'public/data/story.json'), 'utf8'))
} catch (e) {
  console.error('check-values: cannot read public/data/values.json or story.json: ' + e.message)
  process.exit(1)
}
const missing = [...used.keys()].filter((k) => !(k in values) || String(values[k]).trim() === '')
if (missing.length) {
  console.error(`\ncheck-values: ${missing.length} value key(s) used by the site are missing from public/data/values.json:`)
  for (const k of missing.sort()) console.error(`  ${k}   (${[...new Set(used.get(k))].join(', ')})`)
  console.error('\nRegenerate with `npm run data` (src/viz/export_web_data.py) or fix the key name in the source.\n')
  process.exit(1)
}
console.log(`check-values: ${used.size} keys used, all present.`)
