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
            //disabled={isGenerating}
            rows={1}
          />
          <button id="send" onClick={onSend} title="Send Message">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <path fill-rule="evenodd" clip-rule="evenodd" d="M8.73167 5.77133L5.66953 9.91436C4.3848 11.6526 3.74244 12.5217 4.09639 13.205C4.10225 13.2164 4.10829 13.2276 4.1145 13.2387C4.48945 13.9117 5.59888 13.9117 7.81775 13.9117C9.05079 13.9117 9.6673 13.9117 10.054 14.2754L10.074 14.2946L13.946 9.72466L13.926 9.70541C13.5474 9.33386 13.5474 8.74151 13.5474 7.55682V7.24712C13.5474 3.96249 13.5474 2.32018 12.6241 2.03721C11.7007 1.75425 10.711 3.09327 8.73167 5.77133Z" fill="currentColor"></path> <path opacity="0.5" d="M10.4527 16.4432L10.4527 16.7528C10.4527 20.0374 10.4527 21.6798 11.376 21.9627C12.2994 22.2457 13.2891 20.9067 15.2685 18.2286L18.3306 14.0856C19.6154 12.3474 20.2577 11.4783 19.9038 10.7949C19.8979 10.7836 19.8919 10.7724 19.8857 10.7613C19.5107 10.0883 18.4013 10.0883 16.1824 10.0883C14.9494 10.0883 14.3329 10.0883 13.9462 9.72461L10.0742 14.2946C10.4528 14.6661 10.4527 15.2585 10.4527 16.4432Z" fill="currentColor"></path> </g></svg>
          </button>
        </div>
      </div>
      <p className="disclaimer-text">
      Beta Developmental Version! Unexpected Behaviour or Self-awareness/Self-Consciousness incident/accident may occour! Use it with caution! And this is NOT AGI. Do not use as such!
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