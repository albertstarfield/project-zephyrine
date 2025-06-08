// externalAnalyzer/frontend-face-zephyrine/src/components/KnowledgeTuningPage.jsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import '../styles/components/_knowledgeTuning.css';
import { v4 as uuidv4 } from 'uuid'; // Import uuidv4 for temporary file IDs

// This is the LLM API endpoint for file uploads
const LLM_UPLOAD_API_URL = import.meta.env.VITE_LLM_FILE_UPLOAD_API_URL || 'http://localhost:11434/v1/files';
// This is your backend endpoint for saving/fetching metadata about files
const METADATA_API_URL = '/api/v1/files';

// Allowed file types based on common LLM fine-tuning support and specific MIME types
const ALLOWED_MIME_TYPES = [
  'text/plain',              // .txt
  'text/csv',                // .csv
  'text/tab-separated-values', // .tsv
  'application/jsonl',       // .jsonl
  'application/x-parquet',   // .parquet binary format
  // Common spreadsheet formats (will be sent as binary/base64)
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
  'application/vnd.ms-excel', // .xls
];

// Helper to map file types to common extensions for display
const FILE_TYPE_MAP = {
    'text/plain': 'txt',
    'text/csv': 'csv',
    'text/tab-separated-values': 'tsv',
    'application/jsonl': 'jsonl',
    'application/x-parquet': 'parquet',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
    'application/vnd.ms-excel': 'xls',
};

// Function to determine if a file is text-based or binary
const isTextFile = (mimeType) => mimeType.startsWith('text/') || mimeType === 'application/jsonl';

// Function to get file extension from filename
const getFileExtension = (filename) => filename.split('.').pop().toLowerCase();

const KnowledgeTuningPage = () => {
  const { user } = useAuth();
  const fileInputRef = useRef(null);

  // filesToQueue: files selected by user, waiting to be uploaded
  const [filesToQueue, setFilesToQueue] = useState([]);
  // uploadedFiles: files that have been sent to the backend, persistent
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isUploadingGlobal, setIsUploadingGlobal] = useState(false); // Global upload state for button
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [uploadError, setUploadError] = useState(null);
  const [historyError, setHistoryError] = useState(null);
  const [uploadMessage, setUploadMessage] = useState('');
  const [isDragging, setIsDragging] = useState(false);

  // Function to fetch file history from your backend (metadata)
  const fetchFileHistory = useCallback(async () => {
    if (!user?.id) {
      setLoadingHistory(false);
      return;
    }
    setLoadingHistory(true);
    setHistoryError(null);
    try {
      const response = await fetch(`${METADATA_API_URL}?userId=${user.id}`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to fetch file history metadata.');
      }
      const result = await response.json();
      setUploadedFiles(result.data.map(file => ({
          ...file,
          status: file.status ? file.status.toLowerCase() : 'unknown',
          progress: 100 // Assume fetched files are fully processed/uploaded
      })) || []);
    } catch (err) {
      console.error('Error fetching file history:', err);
      setHistoryError(err.message);
    } finally {
      setLoadingHistory(false);
    }
  }, [user]);

  // Fetch file history on component mount or when user changes
  useEffect(() => {
    fetchFileHistory();
  }, [fetchFileHistory]);

  // Handle file selection from input or drop zone
  const handleFileChange = (fileList) => {
    setUploadError(null);
    setUploadMessage('');
    const newFiles = Array.from(fileList).filter(file => {
      // Validate by MIME type first, then by extension as fallback
      const isValidMime = ALLOWED_MIME_TYPES.includes(file.type);
      const ext = getFileExtension(file.name);
      const isValidExtension = Object.values(FILE_TYPE_MAP).flat().includes(ext);

      if (!isValidMime && !isValidExtension && file.type !== "") { // Allow empty file.type to be checked by extension
        setUploadError(prev => `${prev ? prev + '\n' : ''}Unsupported file type or extension: "${file.name}" (${file.type || 'unknown type'}).`);
        return false;
      }
      return true;
    }).map(file => ({
        // Create a temporary ID for optimistic UI
        id: uuidv4(),
        filename: file.name,
        filetype: file.type || `application/${getFileExtension(file.name)}`, // Fallback for missing MIME type
        status: 'queued',
        uploaded_at: new Date().toISOString(),
        progress: 0,
        actualFile: file // Keep reference to actual File object
    }));

    if (newFiles.length > 0) {
      setFilesToQueue(prev => [...prev, ...newFiles]);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = ''; // Clear input to allow re-selection of same file
    }
  };

  // Function to upload a single file directly to LLM API
  const uploadSingleFileToLLM = useCallback(async (fileToUpload) => {
    return new Promise(async (resolve, reject) => {
      // FileReader is not needed for FormData.append(File)
      // The browser handles reading the file from disk directly for multipart/form-data.

      // Update the status of this specific file in the queue
      setFilesToQueue(prev => prev.map(f => f.id === fileToUpload.id ? { ...f, status: 'uploading', progress: 10 } : f));
      setUploadedFiles(prev => prev.map(f => f.id === fileToUpload.id ? { ...f, status: 'uploading', progress: 10 } : f));
      
      try {
        const formData = new FormData();
        formData.append('purpose', 'fine-tune'); // As per curl example
        formData.append('file', fileToUpload.actualFile); // Append the actual File object

        const API_KEY_OR_TOKEN = import.meta.env.VITE_OPENAI_API_KEY || 'your-ollama-token-if-needed'; // FIXED: process.env -> import.meta.env

        const response = await fetch(LLM_UPLOAD_API_URL, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${API_KEY_OR_TOKEN}`,
            // FormData automatically sets Content-Type: multipart/form-data
          },
          body: formData,
        });

        if (!response.ok) {
          let errorText = await response.text();
          try {
            const errorData = JSON.parse(errorText);
            throw new Error(errorData.message || JSON.stringify(errorData));
          } catch {
            throw new Error(`LLM API upload failed with status ${response.status}: ${errorText}`);
          }
        }

        const llmResult = await response.json();
        const llmFileId = llmResult.id || llmResult.file_id; // Depending on LLM API response

        setUploadMessage(`File "${fileToUpload.filename}" uploaded to LLM API.`);
        // Update status and progress after successful LLM upload
        setFilesToQueue(prev => prev.filter(f => f.id !== fileToUpload.id)); // Remove from queue
        setUploadedFiles(prev => prev.map(f => f.id === fileToUpload.id ? { ...f, status: 'processing', progress: 75 } : f));

        // Now, save metadata to your backend
        const metadataResponse = await fetch(METADATA_API_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: fileToUpload.filename,
            filetype: fileToUpload.filetype,
            userId: user.id,
            llmFileId: llmFileId || null, // Pass the LLM-assigned ID
            status: 'uploaded', // Status from LLM upload success
          }),
        });

        if (!metadataResponse.ok) {
          const metadataErrorData = await metadataResponse.json();
          throw new Error(`Failed to save metadata to your backend: ${metadataErrorData.message || JSON.stringify(metadataErrorData)}`);
        }
        const metadataResult = await metadataResponse.json();
        // Update the file in uploadedFiles with the actual backend ID and final status
        setUploadedFiles(prev => prev.map(f => f.id === fileToUpload.id ? { ...f, ...metadataResult.file, status: 'finished', progress: 100 } : f));
        resolve(metadataResult);

      } catch (err) {
        console.error('Final upload process error:', err);
        setUploadError(prev => `${prev ? prev + '\n' : ''}Failed to upload ${fileToUpload.filename}: ${err.message}`);
        setUploadMessage('');
        // Update the status of the file to 'error'
        setFilesToQueue(prev => prev.filter(f => f.id !== fileToUpload.id)); // Remove from queue
        setUploadedFiles(prev => prev.map(f => f.id === fileToUpload.id ? { ...f, status: 'error', progress: 0 } : f));
        reject(err);
      }
    });
  }, [user]);


  // Function to initiate upload for all files in the queue
  const handleUploadAllQueuedFiles = async () => {
    if (filesToQueue.length === 0 || isUploadingGlobal) return;
    setIsUploadingGlobal(true);
    setUploadError(null);
    setUploadMessage('Starting uploads...');

    // Add optimistic entries to uploadedFiles for display and clear queue
    const currentQueue = filesToQueue.map(file => ({
        ...file,
        id: uuidv4(), // Assign a fresh temporary ID
        status: 'queued',
        progress: 0
    }));
    setUploadedFiles(prev => [...currentQueue, ...prev]);
    setFilesToQueue([]); // Clear the internal queue state

    for (const fileItem of currentQueue) { // Iterate over the newly optimistic list
        try {
            await uploadSingleFileToLLM(fileItem);
        } catch (error) {
            // Error handling is inside uploadSingleFileToLLM
        }
    }
    setIsUploadingGlobal(false);
    setUploadMessage('All selected files processed.');
    // Re-fetch history to ensure final consistency with DB
    // fetchFileHistory(); // This is called after each file upload, but a final check is good.
  };

  // Drag and Drop Handlers
  const handleDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((event) => {
    event.preventDefault();
    setIsDragging(false);
    handleFileChange(event.dataTransfer.files);
  }, [handleFileChange]);

  // Helper for status pill rendering
  const getStatusPillClass = (status) => {
    switch (status) {
      case 'queued': return 'queued';
      case 'uploading': return 'uploading';
      case 'finished': return 'finished';
      case 'processing': return 'processing';
      case 'error': return 'error';
      default: return '';
    }
  };

  return (
    <div className="knowledge-tuning-container">
      <img src="/img/ProjectZephy023LogoRenewal.png" alt="Project Zephyrine Logo" className="tuning-page-logo" />

      <div className="tuning-header">
        <h1>Skill Fine-Tuning</h1>
        <p>Upload files to fine-tune Zephyrine's knowledge base.</p>
        <span>
            Supported formats: {
                Object.values(FILE_TYPE_MAP).map(ext => Array.isArray(ext) ? ext.join(', ').toUpperCase() : ext.toUpperCase()).join(', ')
            }
        </span>
      </div>

      <div
        className={`drop-zone ${isDragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={(e) => handleFileChange(e.target.files)}
          disabled={isUploadingGlobal || !user?.id}
          // Accept only specified MIME types or common extensions for initial filtering
          accept={ALLOWED_MIME_TYPES.map(mime => mime).join(',') + ',' + Object.values(FILE_TYPE_MAP).flat().map(ext => `.${ext}`).join(',')}
          webkitdirectory="true" // Enable folder selection
          directory="true" // Broader compatibility for folder selection
          multiple // Allow multiple files selection
          style={{ display: 'none' }}
        />
        <div className="drop-zone-prompt">
          <p>Drag & Drop files or a folder here, or click to browse</p>
          {filesToQueue.length > 0 && (
            <p className="selected-file-info">
              {filesToQueue.length} file(s) in queue: {filesToQueue.map(f => f.filename).join(', ')}
            </p>
          )}
          <button
            onClick={(e) => { e.stopPropagation(); handleUploadAllQueuedFiles(); }}
            disabled={filesToQueue.length === 0 || isUploadingGlobal || !user?.id}
            className="browse-button"
          >
            {isUploadingGlobal ? 'Uploading...' : `Upload ${filesToQueue.length > 0 ? filesToQueue.length : ''} File(s)`}
          </button>
        </div>
      </div>

      {uploadError && <div className="error-message">{uploadError}</div>}
      {uploadMessage && <div className="success-message">{uploadMessage}</div>}

      <div className="file-list-container">
        <h2>Uploaded Files History</h2>
        {loadingHistory ? (
          <p className="empty-list-message">Loading file history...</p>
        ) : historyError ? (
          <div className="error-message">{historyError}</div>
        ) : uploadedFiles.length === 0 ? (
          <p className="empty-list-message">No files uploaded yet. Upload files above!</p>
        ) : (
          <table className="file-list-table">
            <thead>
              <tr>
                <th>File Name</th>
                <th>Type</th>
                <th>Status</th>
                <th>Uploaded At</th>
                <th>LLM ID</th>
              </tr>
            </thead>
            <tbody>
              {uploadedFiles.map((file) => (
                <tr key={file.id}>
                  <td>{file.filename}</td>
                  <td>{file.filetype.split('/').pop() || getFileExtension(file.filename)}</td>
                  <td>
                    {(file.status === 'uploading' || file.status === 'queued') && file.progress !== undefined ? (
                        <div className="progress-bar-container">
                            <div
                                className="progress-bar"
                                style={{ width: `${file.progress || 0}%` }}
                            ></div>
                        </div>
                    ) : null}
                    <span className={`status-pill ${getStatusPillClass(file.status)}`}>
                      {file.status}
                    </span>
                  </td>
                  <td>{new Date(file.uploaded_at).toLocaleString()}</td>
                  <td>{file.llm_file_id || 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default KnowledgeTuningPage;