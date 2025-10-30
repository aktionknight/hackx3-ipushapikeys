import React, { useState } from 'react';
import styles from './SearchBar.module.css';

export default function SearchBar({ onSearch }) {
  const [mode, setMode] = useState('name'); // 'name' | 'coords'
  const [name, setName] = useState('');
  const [lat, setLat] = useState('');
  const [lng, setLng] = useState('');

  function handleSubmit(e) {
    e.preventDefault();
    if (mode === 'name') {
      if (!name.trim()) return;
      onSearch && onSearch({ mode: 'name', name: name.trim() });
    } else {
      if (lat === '' || lng === '') return;
      const latitude = Number(lat);
      const longitude = Number(lng);
      if (Number.isNaN(latitude) || Number.isNaN(longitude)) return;
      onSearch && onSearch({ mode: 'coords', latitude, longitude });
    }
  }

  return (
    <div className={styles.container}>
      <form className={styles.form} onSubmit={handleSubmit}>
        <div className={styles.modeRow}>
          <label className={styles.label}>Search by</label>
          <select className={styles.select} value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="name">Place name</option>
            <option value="coords">Latitude/Longitude</option>
          </select>
        </div>

        {mode === 'name' ? (
          <input
            className={styles.input}
            type="text"
            placeholder="e.g., Manali, Himachal Pradesh"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        ) : (
          <div className={styles.coordsRow}>
            <input
              className={styles.input}
              type="number"
              step="any"
              placeholder="Latitude (e.g., 32.239)"
              value={lat}
              onChange={(e) => setLat(e.target.value)}
            />
            <input
              className={styles.input}
              type="number"
              step="any"
              placeholder="Longitude (e.g., 77.189)"
              value={lng}
              onChange={(e) => setLng(e.target.value)}
            />
          </div>
        )}

        <button className={styles.button} type="submit">Search</button>
      </form>
    </div>
  );
}
