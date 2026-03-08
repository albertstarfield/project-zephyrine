import React from "react";

const SystemOverlay = () => {
  // Using span instead of the non-standard overlaytext tag
  return (
    <div id="overlay-info">
      <span>
        {/* Pre-Alpha Developmental Version! Instabilities may occur! */}
      </span>
      {/* <span>Use with caution! -Zephyrine Foundation (2024)</span> */}
    </div>
  );
};

export default SystemOverlay;
