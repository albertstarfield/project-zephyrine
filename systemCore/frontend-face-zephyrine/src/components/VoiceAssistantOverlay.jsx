// externalAnalyzer/frontend-face-zephyrine/src/components/VoiceAssistantOverlay.jsx
import React, { useEffect, useRef, useState } from 'react'; // FIXED: Corrected import syntax
import PropTypes from 'prop-types';
import '../styles/components/_voiceOverlay.css';
import { useVoiceProcessor } from '../hooks/useVoiceProcessor';
import { useAuth } from '../contexts/AuthContext';

import { MessageSquare, Mic, StopCircle } from 'lucide-react';

const VoiceAssistantOverlay = ({ isVisible, toggleVoiceAssistant, onSendMessage }) => {
    const { user } = useAuth();
    // Destructure isProcessing from useVoiceProcessor
    const { isRecording, isSpeaking, isProcessing, transcript, activateMic, deactivateMic, isMicActive, error: voiceProcessorError } = useVoiceProcessor(); 
    const [localTranscript, setLocalTranscript] = useState('');
    const lastTranscriptRef = useRef('');

    const [displayError, setDisplayError] = useState(null);

    useEffect(() => {
        setLocalTranscript(transcript);
        if (transcript && !isSpeaking && transcript !== lastTranscriptRef.current) {
            // Only update ref if transcript is final and not speaking
            lastTranscriptRef.current = transcript;
        }
    }, [transcript, isSpeaking]);

    useEffect(() => {
        if (voiceProcessorError) {
            setDisplayError(voiceProcessorError);
        } else {
            setDisplayError(null);
        }
    }, [voiceProcessorError]);

    const handleSendTranscript = () => {
        if (localTranscript && localTranscript.trim()) {
            onSendMessage(localTranscript);
            setLocalTranscript('');
            deactivateMic(); // Deactivate mic after sending
            lastTranscriptRef.current = '';
        }
    };

    /**
     * FIXED: This useEffect now explicitly manages mic state based on isVisible and user.
     * The `isMicActive` state from useVoiceProcessor is *read* inside the effect, but
     * is *not* a direct dependency of this effect. This breaks the update loop.
     * `activateMic` and `deactivateMic` are stable useCallback functions from the hook.
     */
    useEffect(() => {
        // Read the current state of the mic from the hook directly
        const micIsCurrentlyActive = isMicActive; 
        // Determine if the mic *should* be active based on overlay visibility and user login
        const shouldMicBeActive = isVisible && user;

        // If the mic should be active, but isn't currently, activate it
        if (shouldMicBeActive && !micIsCurrentlyActive) {
            console.log("VoiceAssistantOverlay: Activating mic.");
            activateMic();
        // If the mic should NOT be active, but is currently, deactivate it
        } else if (!shouldMicBeActive && micIsCurrentlyActive) {
            console.log("VoiceAssistantOverlay: Deactivating mic.");
            deactivateMic();
        }
        // Dependencies for this effect are now only 'isVisible', 'user',
        // and the stable 'activateMic'/'deactivateMic' callbacks.
        // `isMicActive` is intentionally omitted from the dependency array to break the loop.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isVisible, user, activateMic, deactivateMic]);


    // Determine the animation class for audio pills based on voice processing state
    const pillAnimationClass = isRecording ? 'listening' : isSpeaking ? 'speaking' : isProcessing ? 'processing' : 'idle';

    return (
        <div className={`voice-assistant-overlay ${isVisible ? 'visible' : ''}`}>
            <div className="overlay-content">
                {/* Button to close the voice assistant overlay */}
                <button className="close-overlay-button" onClick={toggleVoiceAssistant}>
                    <StopCircle size={24} /> {/* Lucide icon for closing */}
                </button>

                {/* Zephy Logo */}
                <img
                    src="/img/ProjectZephy023LogoRenewal.png" // Path to your Zephy logo
                    alt="Zephy Logo"
                    className="overlay-logo" // Add a class for styling
                />

                {/* Main visual elements */}
                <MessageSquare className="main-icon" size={48} />
                <h2 className="overlay-title">Voice Assistant</h2>
                {/* Display current status or errors */}
                <p className="overlay-status">
                    {displayError ? displayError : (isRecording ? 'Listening...' : isSpeaking ? 'Zephyrine is speaking...' : isProcessing ? 'Processing...' : 'Tap Mic to Speak')}
                </p>

                {/* Audio Visualization Pills Container */}
                <div className="audio-pills-container">
                    {/* Render 5 audio pills with dynamic animation classes and staggered delays */}
                    {[...Array(5)].map((_, index) => (
                        <div
                            key={index}
                            className={`audio-pill ${pillAnimationClass}`}
                            style={{ '--pill-index': index }} // Pass index as CSS variable for animation staggering
                        ></div>
                    ))}
                </div>

                {/* Display area for the transcribed text */}
                <div className="transcript-display">
                    {localTranscript || (isRecording ? 'Waiting for your voice...' : 'Say something...')}
                </div>

                {/* Action buttons (Mic control and Send message) */}
                <div className="overlay-actions">
                    <button
                        className="mic-button"
                        onClick={isMicActive ? deactivateMic : activateMic}
                        disabled={!user} // Disable mic if no user is authenticated
                    >
                        <Mic size={36} /> {/* Lucide icon for microphone */}
                        <span>{isMicActive ? 'Stop Listening' : 'Start Mic'}</span>
                    </button>
                    <button
                        className="send-button"
                        onClick={handleSendTranscript}
                        // Disable send button if transcript is empty or only whitespace
                        disabled={!localTranscript || !localTranscript.trim()}
                    >
                        <MessageSquare size={36} /> {/* Lucide icon for sending message */}
                        <span>Send Message</span>
                    </button>
                </div>
            </div>
        </div>
    );
};

/**
 * PropTypes for the VoiceAssistantOverlay component to enforce prop types.
 */
VoiceAssistantOverlay.propTypes = {
    isVisible: PropTypes.bool.isRequired, // Controls if the overlay is visible
    toggleVoiceAssistant: PropTypes.func.isRequired, // Callback to toggle overlay visibility
    onSendMessage: PropTypes.func.isRequired, // Callback to send message to parent (App.jsx)
};

export default VoiceAssistantOverlay;
