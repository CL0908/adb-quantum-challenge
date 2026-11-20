/** JS mirror of the theme.css dark tokens, for canvas / three.js code that cannot read CSS variables. Keep in sync with theme.css. */
export const DARK = {
  bg: '#0D1520', bg2: '#121C2A', ink: '#EAEFF5',
  teal: '#4DBBAB', terra: '#E8795C', navyLight: '#86A9DC', neutral: '#8C9BB2', sand: '#E9D7AE',
  terrainA: '#16253A',
} as const
export const LIGHT = {
  bg: '#F3F1EC', ink: '#111413', teal: '#0E8074', terra: '#C8553A', navyLight: '#2F6AB0', neutral: '#8A8E98', sand: '#AD7A00', navy: '#1F3557',
} as const
export const NAVY = '#1F3557'
/** '#RRGGBB' -> 'r,g,b' for rgba() strings. */
export const rgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)).join(',')
