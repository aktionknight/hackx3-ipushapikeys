import React from 'react';
import { useMemo, useState, useCallback, useRef } from 'react'
import styles from './Visualizer.module.css'

function ToggleTabs({ activeTab, onChange }) {
  return (
    <div className={styles.tabs}>
      <button
        className={activeTab === 'heatmap' ? styles.tabActive : styles.tab}
        onClick={() => onChange('heatmap')}
      >
        Risk Heatmap
      </button>
      <button
        className={activeTab === 'terrain' ? styles.tabActive : styles.tab}
        onClick={() => onChange('terrain')}
      >
        3D Terrain Render
      </button>
    </div>
  )
}

function HeatmapContainer() {
  const [points, setPoints] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const onUpload = useCallback(async (e) => {
    const file = e.target.files && e.target.files[0]
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch('http://127.0.0.1:8000/analyze/', {
        method: 'POST',
        body: form
      })
      if (!res.ok) throw new Error(`Backend error ${res.status}`)
      const json = await res.json()
      const pts = Array.isArray(json?.risk_heatmap) ? json.risk_heatmap : []
      setPoints(pts)
    } catch (err) {
      setError(err.message || 'Failed to fetch heatmap')
    } finally {
      setLoading(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }, [])

  // compute bbox
  const bbox = useMemo(() => {
    if (!points.length) return null
    let minLat = Infinity, maxLat = -Infinity, minLon = Infinity, maxLon = -Infinity
    for (const p of points) {
      minLat = Math.min(minLat, p.latitude)
      maxLat = Math.max(maxLat, p.latitude)
      minLon = Math.min(minLon, p.longitude)
      maxLon = Math.max(maxLon, p.longitude)
    }
    // pad bbox a bit
    const padLat = (maxLat - minLat) * 0.1 || 0.01
    const padLon = (maxLon - minLon) * 0.1 || 0.01
    return { minLat: minLat - padLat, maxLat: maxLat + padLat, minLon: minLon - padLon, maxLon: maxLon + padLon }
  }, [points])

  const project = useCallback((lat, lon) => {
    if (!bbox) return { x: 0, y: 0 }
    const x = (lon - bbox.minLon) / (bbox.maxLon - bbox.minLon)
    const y = 1 - (lat - bbox.minLat) / (bbox.maxLat - bbox.minLat)
    return { x: Math.min(1, Math.max(0, x)), y: Math.min(1, Math.max(0, y)) }
  }, [bbox])

  return (
    <div className={styles.heatmap}>
      <div className={styles.heatmapToolbar}>
        <input ref={inputRef} type="file" accept="image/*" onChange={onUpload} />
        {loading && <span className={styles.badge}>Analyzing…</span>}
        {error && <span className={styles.error}>{error}</span>}
      </div>
      <div className={styles.mapStage}>
        {!points.length && !loading && (
          <div className={styles.placeholderText}>Upload a terrain image to generate risk heatmap</div>
        )}
        {points.map((p, idx) => {
          const { x, y } = project(p.latitude, p.longitude)
          const color = p.risk_level >= 4 ? '#ef4444' : p.risk_level === 3 ? '#f59e0b' : '#22c55e'
          return (
            <div
              key={idx}
              className={styles.point}
              style={{ left: `${x * 100}%`, top: `${y * 100}%`, backgroundColor: color }}
              title={`${p.risk_level}/5 - ${p.description}`}
            />
          )
        })}
      </div>
    </div>
  )
}

function TerrainPlaceholder() {
  return (
    <div className={styles.terrain}>
      <div className={styles.placeholderText}>3D Viewer Placeholder (Zoom/Rotate)</div>
    </div>
  )
}

export default function Visualizer({ activeTab, onChangeTab }) {
  const content = useMemo(() => {
    return activeTab === 'terrain' ? <TerrainPlaceholder /> : <HeatmapContainer />
  }, [activeTab])

  return (
    <div className={styles.container}>
      <div className={styles.headerRow}>
        <h2 className={styles.heading}>Visualizer</h2>
        <ToggleTabs activeTab={activeTab} onChange={onChangeTab} />
      </div>
      <div className={styles.content}>{content}</div>
    </div>
  )
}