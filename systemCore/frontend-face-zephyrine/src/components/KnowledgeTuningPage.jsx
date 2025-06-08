import React, { useState, useRef, useCallback } from 'react';
import '../styles/components/_knowledgeTuning.css';

const KnowledgeTuningPage = () => {
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileAdd = useCallback((newFiles) => {
    const filesArray = Array.from(newFiles).map(file => ({
      id: `${file.name}-${file.lastModified}`,
      name: file.name,
      size: file.size,
      type: file.type,
      progress: 0,
      status: 'Queued',
      fileObject: file // Keep the actual file object for upload
    }));
    setFiles(prevFiles => [...prevFiles, ...filesArray]);
  }, []);
  
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };
  
  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileAdd(e.dataTransfer.files);
      e.dataTransfer.clearData();
    }
  };

  const onFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileAdd(e.target.files);
    }
  };

  const formatBytes = (bytes, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  return (
    <div className="knowledge-tuning-container">
      <img src="/img/ProjectZephy023LogoRenewal.png" alt="Project Zephyrine Logo" className="tuning-page-logo" />
      
      <div className="tuning-header">
        <h1>Knowledge Tuning</h1>
        <p>Upload documents to fine-tune the model's knowledge base.</p>
        <p>Supported Format: jsonl, csv, tsv, parquet</p>
      </div>

      <div 
        className={`drop-zone ${isDragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="drop-zone-prompt">
          <p>Drag & Drop files here</p>
          <span>or</span>
          <input
            type="file"
            multiple
            ref={fileInputRef}
            onChange={onFileSelect}
            style={{ display: 'none' }}
          />
          <button onClick={() => fileInputRef.current.click()} className="browse-button">
            Browse Files
          </button>
        </div>
      </div>

      <div className="file-list-container">
        <h2>File Queue</h2>
        <table className="file-list-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Size</th>
              <th>Type</th>
              <th>Upload Progress</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {files.length > 0 ? (
              files.map(file => (
                <tr key={file.id}>
                  <td data-label="Filename">{file.name}</td>
                  <td data-label="Size">{formatBytes(file.size)}</td>
                  <td data-label="Type">{file.type || 'N/A'}</td>
                  <td data-label="Upload Progress">
                    <div className="progress-bar-container">
                      <div 
                        className="progress-bar" 
                        style={{ width: `${file.progress}%` }}
                      ></div>
                    </div>
                    <span>{file.progress}%</span>
                  </td>
                  <td data-label="Status">
                    <span className={`status-pill ${file.status.toLowerCase()}`}>
                      {file.status}
                    </span>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="5" className="empty-list-message">No files have been added.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default KnowledgeTuningPage;