import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './home.css'

const still = new URLSearchParams(location.search).get('still')
const root = createRoot(document.getElementById('root')!)
if (still) {
  document.documentElement.style.background = 'transparent'
  document.body.style.background = 'transparent'
  import('./Still').then(({ default: Still }) => root.render(<Still name={still} />))
} else {
  import('./Home').then(({ default: Home }) => root.render(<StrictMode><Home /></StrictMode>))
}
