import { useEffect, useState } from 'react'
import type { Grid } from '../lib/model'
import type { MapData } from '../lib/map'

/** Data files live at the site root (/data), whichever page loads them. */
export const dataUrl = (f: string) => `/data/${f}`

export function useData<T>(file: string): T | null {
  const [d, setD] = useState<T | null>(null)
  useEffect(() => {
    let live = true
    fetch(dataUrl(file))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((j: T) => live && setD(j))
      .catch(() => undefined)
    return () => {
      live = false
    }
  }, [file])
  return d
}

export interface Source { id: string; label: string; kind: 'eq' | 'rain'; scope: 'KG' | 'TJ' | 'both'; rate: number }
export interface Pool { id: string; label: string; sub: string; n: number; edges: number; bond: number | null; seconds: number; method: string }
export interface E1Cell { rdpo: number; mc: number; best: number; coverage: number; seconds: number }
export interface Solution {
  budget: number
  scenarios: string[]
  sources: Source[]
  term_years: number
  margins: Record<'KG' | 'TJ', { attach: number; exhaust: number }>
  pools: Pool[]
  e1: Record<string, Record<string, E1Cell>>
  carec_edges: [string, string][]
  carec_order: string[]
}
export interface Story {
  quantum: { n: number; s_star: number; ci: [number, number] | null; classical_s: number; quantum_s: number }[]
}
export interface RunResult {
  estimate: number; ci: [number, number]; exact: number | null; money: boolean; paths_used: number; seconds?: number
  mc: { estimate: number; ci: [number, number]; hits: number }
}
export type { Grid, MapData }
