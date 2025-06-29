import React from "react";
import "../styles/SystemInfo.css";
import { useSystemInfo } from "../hooks/useSystemInfo"; // 1. Import the hook


const SystemInfo = () => { // 2. Remove the systemInfo prop
  const { // 3. Call the hook inside the component
    cpuUsage,
    cpuFree,
    cpuCount,
    threadsUtilized,
    freeMem,
    totalMem,
    os,
  } = useSystemInfo();


  return (
    <div className="system-info-panel">
      <h3>System Information</h3>
      <div className="info-grid">
        <div className="info-item">
          <span className="info-label">OS:</span>
          <span className="info-value">{os}</span>
        </div>
        <div className="info-item">
          <span className="info-label">CPU Usage:</span>
          <span className="info-value">{(cpuUsage * 100).toFixed(1)}%</span>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${cpuUsage * 100}%` }}
            ></div>
          </div>
        </div>
        <div className="info-item">
          <span className="info-label">CPU Free:</span>
          <span className="info-value">{(cpuFree * 100).toFixed(1)}%</span>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${cpuFree * 100}%` }}
            ></div>
          </div>
        </div>
        <div className="info-item">
          <span className="info-label">CPU Cores:</span>
          <span className="info-value">{cpuCount}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Threads Used:</span>
          <span className="info-value">
            {threadsUtilized} / {cpuCount}
          </span>
        </div>
        <div className="info-item">
          <span className="info-label">Memory:</span>
          <span className="info-value">
            {freeMem.toFixed(1)} GB / {totalMem} GB
          </span>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${(freeMem / totalMem) * 100}%` }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemInfo;
