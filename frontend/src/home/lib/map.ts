export interface MapData {
  bbox: number[]
  countries: Record<string, number[][][]>
  zone: number[][][]
  quakes: number[][] // [lat, lon, magnitude, year, reachesBoth]
}
