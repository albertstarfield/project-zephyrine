import React from 'react';
import PropTypes from 'prop-types';
import '../styles/components/_imageModal.css';

const ImageModal = ({ imageUrl, onClose }) => {
  // Prevent clicks inside the image from closing the modal
  const handleContentClick = (e) => {
    e.stopPropagation();
  };

  return (
    <div className="image-modal-overlay" onClick={onClose}>
      <div className="image-modal-content" onClick={handleContentClick}>
        <button className="image-modal-close" onClick={onClose}>
          &times;
        </button>
        <img src={imageUrl} alt="Zoomed preview" />
      </div>
    </div>
  );
};

ImageModal.propTypes = {
  imageUrl: PropTypes.string.isRequired,
  onClose: PropTypes.func.isRequired,
};

export default ImageModal;