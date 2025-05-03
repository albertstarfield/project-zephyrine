import { useState, useEffect } from 'react';

export function useSystemInfo() {
  const [systemInfo, setSystemInfo] = useState({
    username: '',
    assistantName: "Adelaide Zephyrine Charlotte",
    cpuUsage: 0,
    cpuFree: 0,
    cpuCount: 0,
    threadsUtilized: 0,
    freeMem: 0,
    totalMem: 0,
    os: '',
  });

  useEffect(() => {
    // Initial static info
    setSystemInfo((prev) => ({
      ...prev,
      username: "User", // Placeholder
      cpuCount: navigator.hardwareConcurrency || 4,
      os: navigator.platform,
      totalMem: 8, // Simulated 8GB
    }));

    // Interval for dynamic info simulation
    const interval = setInterval(() => {
      setSystemInfo((prev) => {
        const cpuUsage = Math.random() * 0.5;
        return {
          ...prev,
          cpuUsage: cpuUsage,
          cpuFree: 1 - cpuUsage,
          freeMem: Math.random() * 4 + 2, // Simulate 2-6GB free
          threadsUtilized: Math.floor(
            Math.random() * (prev.cpuCount || 4)
          ),
        };
      });
    }, 2000);

    // Cleanup interval on unmount
    return () => clearInterval(interval);
  }, []); // Run only once on mount

  return systemInfo;
}
