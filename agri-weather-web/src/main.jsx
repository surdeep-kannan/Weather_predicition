import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
// Removed standalone imports/rendering for components now managed by App.jsx's routing

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)