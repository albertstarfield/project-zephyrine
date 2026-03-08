import React, { useState, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import '../styles/components/_settingsModal.css';

// --- START: Configuration Metadata ---
const settingMetadata = {
    MEMORY_SIZE: { desc: "Number of recent interactions to keep in memory for context.", min: 1, max: 20, step: 1 },
    ANSWER_SIZE_WORDS: { desc: "Target word count for generated answers.", min: 128, max: 16384, step: 128 },
    TOPCAP_TOKENS: { desc: "Absolute maximum token limit for any LLM call.", min: 1024, max: 32768, step: 256 },
    DEFAULT_LLM_TEMPERATURE: { desc: "Controls randomness. Higher is more creative, lower is more deterministic. Max: 1.0", min: 0, max: 1.0, step: 0.05 },
    RAG_FILE_INDEX_COUNT: { desc: "Max number of indexed files to use as context for a response. Max: 10", min: 0, max: 10, step: 1 },
    RAG_URL_COUNT: { desc: "Max number of crawled URLs to use as context for a response. Max: 10", min: 0, max: 10 },
    FUZZY_SEARCH_THRESHOLD: { desc: "Similarity score (0-100) for fallback history search. Higher is stricter. Max: 85", min: 50, max: 85, step: 1 },
    ENABLE_FILE_INDEXER: { desc: "Enable or disable the background service that indexes uploaded files." },
    ENABLE_PROACTIVE_RE_REFLECTION: { desc: "Allow the AI to spontaneously 're-remember' and reflect on old memories." },
    PROACTIVE_RE_REFLECTION_CHANCE: { desc: "The probability (0.0-1.0) that the AI will trigger a proactive re-reflection.", min: 0, max: 1.0, step: 0.05 },
    MIN_AGE_FOR_RE_REFLECTION_DAYS: { desc: "How many days old a memory must be before it can be proactively re-reflected.", min: 1, max: 30, step: 1 },
    ENABLE_SELF_REFLECTION: { desc: "Enable the primary self-reflection process on recent conversations." },
    SELF_REFLECTION_MAX_TOPICS: { desc: "The maximum number of topics the AI will try to extract from a conversation.", min: 1, max: 50, step: 1 },
    REFLECTION_BATCH_SIZE: { desc: "Number of reflections to process in a single batch.", min: 1, max: 50, step: 1 },
    DEEP_THOUGHT_RETRY_ATTEMPTS: { desc: "How many times to retry a failed 'deep thought' or complex generation task.", min: 0, max: 10, step: 1 },
    RESPONSE_TIMEOUT_MS: { desc: "Timeout in milliseconds for a multi-step generation process to complete.", min: 5000, max: 60000, step: 1000 },
    AGENTIC_RELAXATION_PERIOD_SECONDS: { desc: "The time period (in seconds) over which the agentic relaxation cycle occurs.", min: 0.1, max: 10, step: 0.1 },
    ACTIVE_CYCLE_PAUSE_SECONDS: { desc: "Brief pause between batches during an active reflection cycle.", min: 0, max: 5, step: 0.1 },
    LLAMA_CPP_VERBOSE: { desc: "Enable verbose logging from the llama.cpp backend." },
    REFINEMENT_MODEL_ENABLED: { desc: "Enable the secondary model to refine generated images." },
    REFINEMENT_STRENGTH: { desc: "How much the refiner model alters the initial image (0.0-1.0).", min: 0, max: 1, step: 0.05 },
    REFINEMENT_CFG_SCALE: { desc: "How closely the refiner model follows the prompt.", min: 1, max: 20, step: 0.5 },
    REFINEMENT_ADD_NOISE_STRENGTH: { desc: "Amount of noise to add before refinement, creating variation.", min: 0, max: 5, step: 0.1 },
    ENABLE_DB_SNAPSHOTS: { desc: "Automatically create periodic backups of the database." },
    DB_SNAPSHOT_INTERVAL_MINUTES: { desc: "How often (in minutes) to create a database snapshot.", min: 1, max: 1440, step: 1 },
    DB_SNAPSHOT_RETENTION_COUNT: { desc: "How many of the most recent snapshots to keep.", min: 1, max: 10, step: 1 },
    LOG_BATCH_SIZE: { desc: "Number of log entries to write to the database in a single batch.", min: 1, max: 100, step: 1 },
    LOG_FLUSH_INTERVAL_SECONDS: { desc: "How often (in seconds) to force-flush the log queue to the database.", min: 60, max: 7200, step: 60 },
    ENABLE_STELLA_ICARUS_HOOKS: { desc: "Enable hooks for the Stella Icarus (Python and Ada hooks Narrow Intelligence and Control) module." },
    ENABLE_STELLA_ICARUS_DAEMON: { desc: "Enable the background daemon for the Stella Icarus (Python and Ada hooks Narrow Intelligence and Control) module." },
    INSTRUMENT_STREAM_RATE_HZ: { desc: "Rate (in Hz) to stream instrument bridge.", min: 1, max: 60, step: 1 },
};

const settingCategories = [
  { title: "UI Theme", emoji: "🎨", keys: ["THEME"] },
  { title: "General", emoji: "⚙️", keys: ["MEMORY_SIZE", "ANSWER_SIZE_WORDS", "TOPCAP_TOKENS", "DEFAULT_LLM_TEMPERATURE"] },
  { title: "Augmented Indexing", emoji: "📚", keys: ["ENABLE_FILE_INDEXER", "RAG_FILE_INDEX_COUNT", "RAG_URL_COUNT", "FUZZY_SEARCH_THRESHOLD"] },
  // START: Updated Category Title
  { title: "Augmented Self Improvement", emoji: "🧠", keys: ["ENABLE_SELF_REFLECTION", "ENABLE_PROACTIVE_RE_REFLECTION", "PROACTIVE_RE_REFLECTION_CHANCE", "MIN_AGE_FOR_RE_REFLECTION_DAYS", "SELF_REFLECTION_MAX_TOPICS", "REFLECTION_BATCH_SIZE"] },
  // END: Updated Category Title
  { title: "Image Generation", emoji: "🖼️", keys: ["REFINEMENT_MODEL_ENABLED", "REFINEMENT_STRENGTH", "REFINEMENT_CFG_SCALE", "REFINEMENT_ADD_NOISE_STRENGTH"] },
  { title: "Lang Model Backend", emoji: "🚀", keys: ["LLAMA_CPP_N_GPU_LAYERS", "LLAMA_CPP_N_CTX", "LLAMA_CPP_VERBOSE"] },
  { title: "Stella Icarus Accelerator", emoji: "✈️", keys: ["ENABLE_STELLA_ICARUS_DAEMON", "ENABLE_STELLA_ICARUS_HOOKS", "INSTRUMENT_STREAM_RATE_HZ"] },
  { title: "System & DB", emoji: "💾", keys: ["ENABLE_DB_SNAPSHOTS", "DB_SNAPSHOT_INTERVAL_MINUTES", "DB_SNAPSHOT_RETENTION_COUNT", "LOG_BATCH_SIZE", "LOG_FLUSH_INTERVAL_SECONDS"] },
  { title: "Advanced", emoji: "🔬", keys: ["DEEP_THOUGHT_RETRY_ATTEMPTS", "RESPONSE_TIMEOUT_MS", "AGENTIC_RELAXATION_PERIOD_SECONDS", "ACTIVE_CYCLE_PAUSE_SECONDS"] },
];
// --- END: Configuration Metadata ---


const SettingsModal = ({ isVisible, onClose, onApply, isCriticalMission, onForceThemeChange, currentTheme }) => {
  const [config, setConfig] = useState(null);
  const [initialConfig, setInitialConfig] = useState(null);
  const [error, setError] = useState(null);
  const [showRestartConfirm, setShowRestartConfirm] = useState(false);
  const [forcedTheme, setForcedTheme] = useState(currentTheme);
  const [wingModeOverride, setWingModeOverride] = useState(false);

  // Effect for Wing Mode state
  useEffect(() => {
    try {
      const storedValue = window.localStorage.getItem("wingModeOverride") === 'true';
      setWingModeOverride(storedValue);
    } catch (e) { console.error(e); }
  }, []);

  const handleWingModeToggle = (e) => {
    const isEnabled = e.target.checked;
    setWingModeOverride(isEnabled);
    try {
      window.localStorage.setItem("wingModeOverride", isEnabled);
    } catch (err) {
      console.error("Failed to save wing mode override to localStorage", err);
    }
  };

  const fetchConfig = useCallback(async () => {
    setError(null);
    try {
      const response = await fetch('/ZephyCortexConfig');
      if (!response.ok) {
        throw new Error(`Service Unavailable`);
      }
      const data = await response.json();
      setConfig(data);
      setInitialConfig(data);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    if (isVisible) {
      fetchConfig();
    }
  }, [isVisible, fetchConfig]);

  const handleValueChange = (key, value) => {
    setConfig(prevConfig => ({
      ...prevConfig,
      [key]: value
    }));
  };

  const handleApplyClick = () => {
    if (isCriticalMission) {
      setShowRestartConfirm(true);
    } else {
      applyAndRestart();
    }
  };
  
  const applyAndRestart = async () => {
      if (!config || !initialConfig) return;
  
      const changedSettings = Object.keys(config).reduce((acc, key) => {
          if (config[key] !== initialConfig[key] && settingMetadata[key]) {
              acc[key] = config[key];
          }
          return acc;
      }, {});
  
      const themeChanged = forcedTheme !== currentTheme;
      
      if (Object.keys(changedSettings).length === 0 && !themeChanged) {
          onClose();
          return;
      }

      onForceThemeChange(forcedTheme);

      try {
          if (Object.keys(changedSettings).length > 0) {
            await onApply(changedSettings);
          }
          alert("Settings applied! A restart of the backend service may be required for some changes to take effect.");
          onClose();
      } catch (err) {
          setError(`Failed to apply settings: ${err.message}`);
      }
  };

  const handleCancel = () => {
    onForceThemeChange(currentTheme); // Revert theme change on cancel
    setConfig(initialConfig);
    onClose();
  };

  const handleForceRestart = () => {
    setShowRestartConfirm(false);
    applyAndRestart();
  };
    const handleThemeChange = (event) => {
    const newTheme = event.target.value;
    setForcedTheme(newTheme);
  };


  if (!isVisible) {
    return null;
  }

  return (
    <div className="settings-modal-overlay" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="settings-modal-header">
            <h2>Settings</h2>
            <div className="warning-icon-placeholder">⚠️</div>
        </div>

        <div className="settings-warning">
            This is already set to it's optimum and to it's max, however if you are dare to challenge the tuning and might find out new optimal tuning, go ahead, but instability or inusability may occour!
        </div>
        
        <div className="settings-content">
          {error && <div className="settings-error">Error fetching config: {error}</div>}
          {!config && !error && <div>Loading settings...</div>}
          
          {config && settingCategories.map(category => {
            const keysInCategory = category.keys.filter(key => key in config || key === "THEME");
            if (keysInCategory.length === 0) return null;

            return (
              <div key={category.title} className="settings-category">
                <h3 className="settings-category-title">{category.emoji} {category.title}</h3>
                {keysInCategory.map(key => {
                  if (key === "THEME") {
                    return (
                      <div key="theme-setting" className="setting-item">
                        <div className="setting-label-group">
                           <label className="setting-label">Theme</label>
                           <span className="setting-description">Force a UI theme regardless of system preference.</span>
                        </div>
                        <div className="setting-control">
                           <select className="theme-dropdown" value={forcedTheme} onChange={handleThemeChange}>
                               <option value="dark">Dark</option>
                               <option value="light">Light</option>
                           </select>
                        </div>
                      </div>
                    );
                  }

                  const meta = settingMetadata[key] || {};
                  const value = config[key];

                  return(
                    <div key={key} className="setting-item">
                      <div className="setting-label-group">
                        <label className="setting-label">{key.replace(/_/g, ' ')}</label>
                        <span className="setting-description">{meta.desc || 'No description available.'}</span>
                      </div>
                      <div className="setting-control">
                        {typeof value === 'boolean' ? (
                          <label className="switch">
                            <input type="checkbox" checked={value} onChange={(e) => handleValueChange(key, e.target.checked)} />
                            <span className="slider round"></span>
                          </label>
                        ) : typeof value === 'number' ? (
                          <div className="slider-container">
                            <input
                              type="range"
                              min={meta.min !== undefined ? meta.min : 0}
                              max={meta.max !== undefined ? meta.max : value * 2}
                              step={meta.step !== undefined ? meta.step : (Number.isInteger(value) ? 1 : 0.01)}
                              value={value}
                              onChange={(e) => handleValueChange(key, parseFloat(e.target.value))}
                            />
                            <span className="slider-value">{value.toFixed(meta.step < 1 ? 2 : 0)}</span>
                          </div>
                        ) : (
                          <span className="setting-value-text">{String(value)}</span>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            );
          })}
          
          {/* Wing Mode Footer */}
          <div className="wing-mode-footer">
              <div className="wing-mode-title-group">
                  <h4>✈️ 616574686572323738756E7665696C796F757277696E6773 🚀</h4>
                  <p>Hint: Only if you know what you are doing!</p>
              </div>
              <label className="switch">
                  <input type="checkbox" checked={wingModeOverride} onChange={handleWingModeToggle} />
                  <span className="slider round"></span>
              </label>
          </div>

        </div>
        
        <div className="settings-actions">
          <button onClick={handleCancel} className="settings-button cancel">Cancel</button>
          <button onClick={handleApplyClick} className="settings-button apply" disabled={!config}>Apply and Restart</button>
        </div>
      </div>
      
      {showRestartConfirm && (
        <div className="restart-confirm-overlay" onClick={() => setShowRestartConfirm(false)}>
          <div className="restart-confirm-popup" onClick={(e) => e.stopPropagation()}>
            <h3>Warning: Critical Mission in Progress</h3>
            <p>A critical mission is currently running. Restarting the application may cause data loss or other issues. Are you sure you want to continue?</p>
            <div className="restart-confirm-actions">
              <button onClick={() => setShowRestartConfirm(false)} className="settings-button cancel">Cancel</button>
              <button onClick={handleForceRestart} className="settings-button force-restart">Force Restart</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

SettingsModal.propTypes = {
  isVisible: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  onApply: PropTypes.func.isRequired,
  isCriticalMission: PropTypes.bool.isRequired,
  onForceThemeChange: PropTypes.func.isRequired,
  currentTheme: PropTypes.string.isRequired,
};

export default SettingsModal;