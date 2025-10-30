import React from 'react';
import styles from './Navbar.module.css'

function StatusDot({ status }) {
  const isActive = status === 'active'
  return (
    <span className={styles.status}>
      <span className={isActive ? styles.dotActive : styles.dotOffline} />
      <span className={styles.statusText}>{isActive ? 'Active' : 'Offline'}</span>
    </span>
  )
}

export default function Navbar({ projectTitle, status = 'active', alertsCount = 0 }) {
  return (
    <header className={styles.navbar}>
      <div className={styles.left}>
        <span className={styles.title}>{projectTitle}</span>
      </div>
      <div className={styles.right}>
        <StatusDot status={status} />
        <span className={styles.alertsBadge}>Alerts: {alertsCount}</span>
      </div>
    </header>
  )
}