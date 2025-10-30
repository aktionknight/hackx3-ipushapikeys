import React from 'react';
import { useState } from 'react'
import styles from './App.module.css'
import Navbar from './components/Navbar.jsx'
import Visualizer from './components/Visualizer.jsx'
import InsightsPanel from './components/InsightsPanel.jsx'
import SearchBar from './components/SearchBar.jsx'

export default function App() {
  const [activeTab, setActiveTab] = useState('heatmap')
  const [searchParams, setSearchParams] = useState(null)

  function handleSearch(params) {
    setSearchParams(params)
  }

  return (
    <div className={styles.appRoot}>
      <Navbar projectTitle="Landslide Early-Warning Drone" status="active" alertsCount={3} />

      <div className={styles.searchRow}>
        <SearchBar onSearch={handleSearch} />
      </div>

      <main className={styles.mainContent}>
        <section className={styles.leftColumn}>
          <Visualizer activeTab={activeTab} onChangeTab={setActiveTab} searchParams={searchParams} />
        </section>

        <aside className={styles.rightColumn}>
          <InsightsPanel />
        </aside>
      </main>
    </div>
  )
}