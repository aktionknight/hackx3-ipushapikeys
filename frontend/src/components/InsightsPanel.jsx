import React from 'react';
import styles from './InsightsPanel.module.css'

function RiskIndexCard() {
  return (
    <div className={styles.riskCard}>
      <div className={styles.riskHeader}>
        <span className={styles.riskLabel}>Risk Index</span>
        <span className={styles.riskBadgeHigh}>High</span>
      </div>
      <div className={styles.riskScore}>
        <span className={styles.riskValue}>8.5</span>
        <span className={styles.riskOutOf}>/10</span>
      </div>
      <div className={styles.riskSubtext}>Sample status for demo purposes</div>
    </div>
  )
}

function KeyIndicators() {
  return (
    <div className={styles.indicators}>
      <div className={styles.indicatorCard}>
        <div className={styles.indicatorTitle}>Soil Moisture</div>
        <div className={styles.indicatorValue}>55%</div>
        <div className={styles.indicatorBar}>
          <div className={styles.indicatorFillMoisture} style={{ width: '55%' }} />
        </div>
      </div>

      <div className={styles.indicatorCard}>
        <div className={styles.indicatorTitle}>Rainfall</div>
        <div className={styles.indicatorValue}>45 mm/24h</div>
        <div className={styles.indicatorBar}>
          <div className={styles.indicatorFillRain} style={{ width: '45%' }} />
        </div>
      </div>

      <div className={styles.indicatorCard}>
        <div className={styles.indicatorTitle}>Slope Displacement</div>
        <div className={styles.indicatorValue} style={{ color: '#ff6b6b' }}>+5 mm/hr</div>
        <div className={styles.indicatorBar}>
          <div className={styles.indicatorFillDisplacement} style={{ width: '50%' }} />
        </div>
      </div>
    </div>
  )
}

function AlertsLog() {
  const items = [
    { id: 1, text: 'Crack detected at M-04B' },
    { id: 2, text: 'Subsided soil identified' },
    { id: 3, text: 'Surface shift pattern consistent with creep' },
    { id: 4, text: 'Vegetation stress anomaly in sector S-12' },
    { id: 5, text: 'Runoff channel widening observed' },
  ]

  return (
    <div className={styles.alerts}>
      <div className={styles.alertsHeader}>AI Findings / Alerts</div>
      <div className={styles.alertsList}>
        {items.map(item => (
          <div key={item.id} className={styles.alertItem}>
            {item.text}
          </div>
        ))}
      </div>
    </div>
  )
}

export default function InsightsPanel() {
  return (
    <div className={styles.panelRoot}>
      <RiskIndexCard />
      <KeyIndicators />
      <AlertsLog />
    </div>
  )
}