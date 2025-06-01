// ExternalAnalyzer/frontend-face-zephyrine/src/components/InputArea.jsx
import React, { useRef, useEffect } from "react";
import PropTypes from 'prop-types';

const InputArea = ({ inputValue, onInputChange, onSend, onStop, isGenerating }) => {
  const textareaRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      // Ensure scrollHeight is only applied if it's greater than initial to avoid unnecessary jumps
      if (textareaRef.current.scrollHeight > textareaRef.current.clientHeight) {
        textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
      }
    }
  }, [inputValue]); // <<<< CORRECTED: Use inputValue

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      // Call internal handleSend which now uses inputValue
      handleSendInternal(); 
    }
  };

  // Renamed to avoid confusion with onSend prop, and uses inputValue
  const handleSendInternal = () => { 
    if (inputValue && inputValue.trim() && !isGenerating) { // <<<< CORRECTED: Use inputValue
      onSend(inputValue); // <<<< CORRECTED: Use inputValue
    }
  };

  return (
    <div id="form" className={isGenerating ? "running-model" : ""}>
      <div className="input-container">
        <div className="input-field">
          {/* File Upload Button Placeholder */}
          <button className="input-action-button" title="Upload File">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <g id="SVGRepo_bgCarrier" strokeWidth="0"></g>
              <g
                id="SVGRepo_tracerCarrier"
                strokeLinecap="round"
                strokeLinejoin="round"
              ></g>
              <g id="SVGRepo_iconCarrier">
                {" "}
                <path
                  d="M9 12H15"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                ></path>{" "}
                <path
                  d="M12 9L12 15"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                ></path>{" "}
                <path
                  d="M3 10C3 6.22876 3 4.34315 4.17157 3.17157C5.34315 2 7.22876 2 11 2H13C16.7712 2 18.6569 2 19.8284 3.17157C21 4.34315 21 6.22876 21 10V14C21 17.7712 21 19.6569 19.8284 20.8284C18.6569 22 16.7712 22 13 22H11C7.22876 22 5.34315 22 4.17157 20.8284C3 19.6569 3 17.7712 3 14V10Z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                ></path>{" "}
              </g>
            </svg>
          </button>
          {/* Voice Input Button Placeholder */}
          <button className="input-action-button" title="Use Microphone">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <g id="SVGRepo_bgCarrier" strokeWidth="0"></g>
              <g
                id="SVGRepo_tracerCarrier"
                strokeLinecap="round"
                strokeLinejoin="round"
              ></g>
              <g id="SVGRepo_iconCarrier">
                {" "}
                <path
                  d="M12 2C10.3431 2 9 3.34315 9 5V11C9 12.6569 10.3431 14 12 14C13.6569 14 15 12.6569 15 11V5C15 3.34315 13.6569 2 12 2Z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                ></path>{" "}
                <path
                  d="M19 10V11C19 14.866 15.866 18 12 18C8.13401 18 5 14.866 5 11V10"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                ></path>{" "}
                <path
                  d="M12 18V22M12 22H10M12 22H14"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                ></path>{" "}
              </g>
            </svg>
          </button>
          <textarea
            id="input"
            ref={textareaRef}
            value={inputValue} // <<<< CORRECTED: Use inputValue
            onChange={(e) => onInputChange(e.target.value)} // <<<< CORRECTED: Use onInputChange
            onKeyDown={handleKeyDown}
            placeholder="Enter a message here..."
            disabled={isGenerating}
            rows={1}
          />
          {isGenerating ? (
            <button id="stop" onClick={onStop} title="Stop Generation">
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <rect x="6" y="6" width="12" height="12" fill="currentColor" />
              </svg>
            </button>
          ) : (
            <button id="send" onClick={handleSendInternal} disabled={!inputValue || !inputValue.trim()} title="Send Message"> 
              {/* ^^^^ CORRECTED: Use inputValue and handleSendInternal */}
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M20 12L4 4L6 12M20 12L4 20L6 12M20 12H6"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          )}
        </div>
      </div>
      <p className="disclaimer-text">
        Pre-Alpha Developmental Version! Instabilities may occur! Use with
        caution! -Zephyrine Foundation (2025)
      </p>
    </div>
  );
};

// Add PropTypes for type checking
InputArea.propTypes = {
  inputValue: PropTypes.string.isRequired,
  onInputChange: PropTypes.func.isRequired,
  onSend: PropTypes.func.isRequired,
  onStop: PropTypes.func.isRequired, // Assuming this is onStopGeneration from ChatPage
  isGenerating: PropTypes.bool.isRequired,
};

export default InputArea;