import React from "react";

const TitleBar = () => {
  return (
    <header id="titlebar">
      <div id="drag-region">
        <div id="window-title">
          <span>Project Zephy</span>
        </div>
        <div id="window-controls">
          <div className="button" id="min-button">
            <svg width="10" height="10" viewBox="0 0 10 10">
              <path d="M0,5h10v1H0V5z" fill="currentColor" />
            </svg>
          </div>
          <div className="button" id="max-button">
            <svg width="10" height="10" viewBox="0 0 10 10">
              <path d="M0,0v10h10V0H0z M9,9H1V1h8V9z" fill="currentColor" />
            </svg>
          </div>
          <div className="button" id="close-button">
            <svg width="10" height="10" viewBox="0 0 10 10">
              <path
                d="M6.4,5l3.3-3.3c0.4-0.4,0.4-1,0-1.4s-1-0.4-1.4,0L5,3.6L1.7,0.3c-0.4-0.4-1-0.4-1.4,0s-0.4,1,0,1.4L3.6,5L0.3,8.3 c-0.4,0.4-0.4,1,0,1.4C0.5,9.9,0.7,10,1,10s0.5-0.1,0.7-0.3L5,6.4l3.3,3.3C8.5,9.9,8.7,10,9,10s0.5-0.1,0.7-0.3 c0.4-0.4,0.4-1,0-1.4L6.4,5z"
                fill="currentColor"
              />
            </svg>
          </div>
        </div>
      </div>
    </header>
  );
};

export default TitleBar;
