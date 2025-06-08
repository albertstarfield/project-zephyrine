import React, { useRef, useEffect, useState } from "react";
import PropTypes from 'prop-types';

const InputArea = ({
  inputValue,
  onInputChange,
  onSend,
  onStopGeneration,
  isGenerating,
  onFileSelect, // New prop to pass file data up to ChatPage
  selectedFile, // New prop to know if a file is staged
}) => {
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null); // Ref for the hidden file input
  const [isRecording, setIsRecording] = useState(false);
  const recognitionRef = useRef(null);

  // --- Web Speech API Setup ---
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    recognition.onstart = () => setIsRecording(true);
    recognition.onresult = (event) => onInputChange(event.results[0][0].transcript);
    recognition.onerror = (event) => console.error("Speech recognition error", event.error);
    recognition.onend = () => setIsRecording(false);
    recognitionRef.current = recognition;
  }, [onInputChange]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      if (textareaRef.current.scrollHeight > textareaRef.current.clientHeight) {
        textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
      }
    }
  }, [inputValue]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  const handleMicClick = () => {
    if (isRecording) {
      recognitionRef.current?.stop();
    } else {
      recognitionRef.current?.start();
    }
  };

  // --- File Handling ---
  const handleUploadClick = () => {
    // Programmatically click the hidden file input
    fileInputRef.current?.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    console.log("File selected:", file.name, file.type);

    const reader = new FileReader();

    // Handle images
    if (file.type.startsWith("image/")) {
      reader.readAsDataURL(file); // Reads as base64 data URL
      reader.onload = () => {
        onFileSelect({
          name: file.name,
          type: file.type,
          content: reader.result, // The base64 string
        });
      };
    }
    // Handle text files
    else if (file.type === "text/plain") {
      reader.readAsText(file);
      reader.onload = () => {
        onFileSelect({
          name: file.name,
          type: file.type,
          content: reader.result, // The text content
        });
      };
    }
    // Handle other file types if needed
    else {
      alert("Unsupported file type. Please select an image or a .txt file.");
    }
    
    reader.onerror = (error) => {
      console.error("Error reading file:", error);
      alert("Failed to read the file.");
    };
    
    // Clear the input value so the same file can be selected again
    event.target.value = '';
  };


  return (
    <div id="form" className={isGenerating ? "running-model" : ""}>
      <div className="input-container">
        <div className="input-field">
          {/* Hidden file input */}
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange}
            style={{ display: 'none' }}
            accept="image/png, image/jpeg, image/webp, image/gif, text/plain"
          />

          <button 
            className={`input-action-button ${selectedFile ? 'file-selected-highlight' : ''}`} 
            title="Upload File"
            onClick={handleUploadClick}
          >
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><g><path d="M9 12H15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path><path d="M12 9L12 15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path><path d="M3 10C3 6.22876 3 4.34315 4.17157 3.17157C5.34315 2 7.22876 2 11 2H13C16.7712 2 18.6569 2 19.8284 3.17157C21 4.34315 21 6.22876 21 10V14C21 17.7712 21 19.6569 19.8284 20.8284C18.6569 22 16.7712 22 13 22H11C7.22876 22 5.34315 22 4.17157 20.8284C3 19.6569 3 17.7712 3 14V10Z" stroke="currentColor" strokeWidth="1.5"></path></g></svg>
          </button>
          
          <button 
            className={`input-action-button ${isRecording ? 'mic-active' : ''}`} 
            title="Use Microphone"
            onClick={handleMicClick}
          >
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><g><path d="M12 2C10.3431 2 9 3.34315 9 5V11C9 12.6569 10.3431 14 12 14C13.6569 14 15 12.6569 15 11V5C15 3.34315 13.6569 2 12 2Z" stroke="currentColor" strokeWidth="1.5"></path><path d="M19 10V11C19 14.866 15.866 18 12 18C8.13401 18 5 14.866 5 11V10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"></path><path d="M12 18V22M12 22H10M12 22H14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"></path></g></svg>
          </button>
          <textarea
            id="input"
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => onInputChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={selectedFile ? `Ask about: ${selectedFile.name}` : (isRecording ? "Listening..." : "Enter a message here...")}
            disabled={isGenerating}
            rows={1}
          />
          {isGenerating ? (
            <button id="stop" onClick={onStopGeneration} title="Stop Generation">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="6" y="6" width="12" height="12" fill="currentColor" /></svg>
            </button>
          ) : (
            <button id="send" onClick={onSend} disabled={(!inputValue && !selectedFile) || (inputValue && !inputValue.trim() && !selectedFile)} title="Send Message">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M20 12L4 4L6 12M20 12L4 20L6 12M20 12H6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
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

InputArea.propTypes = {
  inputValue: PropTypes.string.isRequired,
  onInputChange: PropTypes.func.isRequired,
  onSend: PropTypes.func.isRequired,
  onStopGeneration: PropTypes.func.isRequired,
  isGenerating: PropTypes.bool.isRequired,
  onFileSelect: PropTypes.func.isRequired,
  selectedFile: PropTypes.object,
};

export default InputArea;