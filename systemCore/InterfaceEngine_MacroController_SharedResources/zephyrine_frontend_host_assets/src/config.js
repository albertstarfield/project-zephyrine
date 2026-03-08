// src/config.js
// If the global var is set (by Zephyrine_Host), use it. 
// Otherwise default to localhost:3001 (for dev mode).
export const FrontendBackendRecieve = window.FrontendBackendRecieve || "http://localhost:11434";