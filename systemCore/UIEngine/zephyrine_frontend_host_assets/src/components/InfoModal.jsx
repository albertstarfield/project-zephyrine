import React from 'react';
import PropTypes from 'prop-types';
import '../styles/components/_infoModal.css';

const InfoModal = ({ message, onClose }) => {
  return (
    <div className="info-modal-overlay" onClick={onClose}>
      <div className="info-modal-content" onClick={(e) => e.stopPropagation()}>
        <p>{message}</p>
        <button onClick={onClose} className="info-modal-close-button">
          OK
        </button>
      </div>
    </div>
  );
};

InfoModal.propTypes = {
  message: PropTypes.string.isRequired,
  onClose: PropTypes.func.isRequired,
};

export default InfoModal;