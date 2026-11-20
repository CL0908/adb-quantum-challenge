export interface TargetCell {
  value: number
  brief_rounded: number
  rdpo_rel_rmse: number
  rdpo_coverage: number
  reps: number
  budget: number
  best_baseline: string
  best_baseline_rel_rmse: number
  mc_rel_rmse: number
}
export type TargetName = 'expected_loss' | 'joint_attachment' | 'exceedance' | 'stop_loss' | 'joint_exhaustion'
export interface Scenario {
  label: string
  desc: string
  delivered: number
  threshold_musd: number
  targets: Record<TargetName, TargetCell>
}
export interface WpTarget {
  target: TargetName
  threshold_musd: number | null
  value: number
  rdpo_rel_rmse: number | null
  rdpo_coverage: number | null
  reps: number | null
  budget: number | null
  mc_rel_rmse: number | null
  twist_rel_rmse: number | null
}
export interface Run {
  file: string
  script: string
  use: string
  commit: string
  date: string
  seed?: number
  reps?: number
}
export interface Pool {
  label: string
  short: string
  n_countries: number
  exact_label: string
  runs: Run[]
  scenarios?: Record<string, Scenario>
  budget?: number
  targets?: WpTarget[]
  delivered?: number | null
  T_moderate?: number
  T_rare?: number
  reps?: number
  curve?: { x: number[]; y: number[] }
}
export interface DashData {
  asof: { date: string; commit: string; file: string }
  pools: Record<'kgtj' | 'carec' | 'ap', Pool>
  scenario_order: string[]
}
export interface GridPoint {
  eq: number
  rain: number
  m: number
  joint_exhaustion: number
  exceed: number
  delivered: number
  joint_attachment: number
  expected_loss: number
  shortfall: number
}
export interface Grid {
  eq_axis: number[]
  rain_axis: number[]
  values: { joint_exhaustion: number[][]; exceed: number[][]; delivered: number[][] }
  brief: GridPoint
  data: GridPoint
}
