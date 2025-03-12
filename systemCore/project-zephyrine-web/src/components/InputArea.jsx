import React, { useRef, useEffect } from "react";

const InputArea = ({ value, onChange, onSend, onStop, isGenerating }) => {
  const textareaRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [value]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (value.trim() && !isGenerating) {
      onSend(value);
    }
  };

  return (
    <div id="form" className={isGenerating ? "running-model" : ""}>
      <div className="input-container">
        <div className="input-field">
          <textarea
            id="input"
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter a message here..."
            disabled={isGenerating}
            rows={1}
          />
          {isGenerating ? (
            <button id="stop" onClick={onStop}>
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
            <button id="send" onClick={handleSend} disabled={!value.trim()}>
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
    </div>
  );
};

export default InputArea;
