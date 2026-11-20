import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

export type Values = Record<string, string>

/** Fetches a JSON file from data/ once; null until it arrives (or if it fails). */
export function useJSON<T>(file: string): T | null {
  const [d, setD] = useState<T | null>(null)
  useEffect(() => {
    let live = true
    fetch(`data/${file}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((j: T) => live && setD(j))
      .catch(() => undefined)
    return () => {
      live = false
    }
  }, [file])
  return d
}

const SUP: Record<string, string> = { '⁻': '−', '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', 'ⁿ': 'n' }

/** Turns unicode superscripts in a value string (3.2×10⁻⁵) into real <sup> elements. */
export function Fancy({ s }: { s: string }) {
  const parts = s.split(/([⁻⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ]+)/)
  return (
    <>
      {parts.map((p, i) =>
        i % 2 ? <sup key={i}>{[...p].map((c) => SUP[c] ?? c).join('')}</sup> : <span key={i}>{p}</span>,
      )}
    </>
  )
}

const Ctx = createContext<Values>({})
export function ValuesProvider({ children }: { children: ReactNode }) {
  const v = useJSON<Values>('values.json')
  return <Ctx.Provider value={v ?? {}}>{children}</Ctx.Provider>
}
export const useValues = () => useContext(Ctx)

/** A number from values.json (never typed by hand). While the file loads it is a blank skeleton, never placeholder text;
 *  a missing key fails the build (frontend/scripts/check-values.mjs). */
export function V({ k }: { k: string }) {
  const v = useValues()
  return v[k] ? <Fancy s={v[k]} /> : <span className="skel" aria-hidden="true" />
}

/** Data behind the small graphics (data/story.json, written by src/viz/export_story.py from results/*.json). */
export interface Story {
  decomp: { kg_solo: number; shared: number; tj_solo: number; rain_kg_or_tj_only: number; rain_both: number; no_payout: number }
  scaling: { ring: [number, number][]; max_bond: number }
  heat: { targets: string[]; paths: Record<string, Record<string, number>>; time: Record<string, Record<string, number>> }
  quantum: { n: number; s_star: number; ci: [number, number] | null; classical_s: number; quantum_s: number }[]
  documents: Record<'proposal' | 'annex' | 'deck', { pages: number; file: string; thumb: string }>
}
export const useStory = () => useJSON<Story>('story.json')
