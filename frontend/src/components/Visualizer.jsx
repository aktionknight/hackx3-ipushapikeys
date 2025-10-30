import React from 'react';
import { useMemo } from 'react'
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

function HeatmapPlaceholder() {
  return (
    <div className={styles.heatmap}>
      <div className={styles.placeholderText}>Heatmap View Placeholder</div>
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
    return activeTab === 'terrain' ? <TerrainPlaceholder /> : <HeatmapPlaceholder />
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