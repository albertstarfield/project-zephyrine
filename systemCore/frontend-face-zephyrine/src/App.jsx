// ExternalAnalyzer/frontend-face-zephyrine/src/App.jsx
import React, { useState, useEffect, useCallback, useRef, Suspense } from "react";
import { Routes, Route, useNavigate, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import { useTheme } from './contexts/ThemeContext';

// Component Imports
import ChatPage from "./components/ChatPage";
import Header from "./components/Header";
import SideBar from "./components/SideBar";
import Auth from "./components/Auth";
import SystemOverlay from "./components/SystemOverlay";
import ImageGenerationPage from "./components/ImageGenerationPage";
import KnowledgeTuningPage from "./components/KnowledgeTuningPage";
import VoiceAssistantOverlay from "./components/VoiceAssistantOverlay";
import RedirectToNewChat from "./components/RedirectToNewChat";
import SystemInfo from "./components/SystemInfo"; // Keep import even if not directly rendered in Routes

// Hook Imports
import { useSystemInfo } from "./hooks/useSystemInfo";
import { useChatHistory } from "./hooks/useChatHistory";
import { useThemedBackground } from './hooks/useThemedBackground'; // NEW: Replaces useStarBackground

// Stylesheet Imports
import "./styles/App.css";
import "./styles/index.css";
import "./styles/Header.css";
import "./styles/ChatInterface.css";
import "./styles/SystemInfo.css";
import 'katex/dist/katex.min.css';


const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL || "ws://localhost:3001";

// AppContent is a functional component that encapsulates the main application logic and routing.
// It is wrapped by AuthProvider and BrowserRouter in main.jsx.
const AppContent = () => {
  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();

  const ws = useRef(null); // useRef to hold the single WebSocket instance across renders

  // --- NEW: Theme and Background Setup ---
  const { theme } = useTheme(); // Get the current theme ('light' or 'dark')
  const backgroundRef = useRef(null); // Create a ref for the background container
  // NEW: Ref for the canvas element itself
  const starsCanvasRef = useRef(null);

  // Pass the new canvasRef to useThemedBackground
  useThemedBackground(backgroundRef, starsCanvasRef); // Pass starsCanvasRef as the second argument now

  useEffect(() => {
    console.log('Current App theme:', theme);
    if (backgroundRef.current) {
      console.log('Background container ref exists:', backgroundRef.current);
    }
  }, [theme, backgroundRef]);

  // useCallback to provide a stable getter for the WebSocket instance, passed to child components/hooks.
  const getWsInstance = useCallback(() => ws.current, []); 

  const { systemInfo } = useSystemInfo(); // Custom hook to get system information
  
  // Custom hook to manage chat history and its interactions with the backend
  const {
    chatHistory,
    isLoadingHistory,
    fetchChatHistory,
    setChatHistory,
    setIsLoadingHistory,
    handleRenameChat, 
    handleDeleteChat
  } = useChatHistory(getWsInstance); // Pass the WebSocket getter to the hook

  // State for controlling the manual open/close of the sidebar
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  // State for controlling the visibility of the Voice Assistant Overlay
  const [isVoiceAssistantVisible, setIsVoiceAssistantVisible] = useState(false); 
  // State for tracking the WebSocket connection status
  const [isConnected, setIsConnected] = useState(false);

  /**
   * useEffect hook for centralized WebSocket connection management.
   * Handles opening, closing, and re-establishing the WebSocket,
   * as well as setting up and cleaning up event listeners.
   */
  useEffect(() => {
    // If no user is authenticated, ensure the WebSocket is closed.
    if (!user?.id) {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        console.log("App.jsx: No user, closing global WebSocket.");
        ws.current.close();
      }
      ws.current = null; // Clear the ref
      setIsConnected(false); // Update connection status
      return; // Exit early if no user
    }

    // If WebSocket is not currently open or doesn't exist, create a new connection.
    if (!ws.current || ws.current.readyState === WebSocket.CLOSED) {
      console.log("App.jsx: Connecting global WebSocket...");
      ws.current = new WebSocket(WEBSOCKET_URL);
      setIsConnected(false); // Set connecting state
    }

    console.log("App.jsx: Attaching WebSocket event handlers.");

    const socket = ws.current;

    // Event handler for successful WebSocket connection
    socket.onopen = () => {
      console.log("App.jsx: Global WebSocket Connected.");
      setIsConnected(true); // Update connection status
      // Fetch initial chat history once connected
      if (typeof fetchChatHistory === 'function') {
        fetchChatHistory(user.id);
      }
    };

    // Event handler for incoming WebSocket messages
    socket.onmessage = (event) => {
      const parsedMessage = JSON.parse(event.data);
      console.log("App.jsx: Global WS Message Received:", parsedMessage.type);

      // Handle different types of messages from the WebSocket server
      switch (parsedMessage.type) {
        case 'chat_history_list':
          // Update the chat history displayed in the sidebar
          if (typeof setChatHistory === 'function') {
            setChatHistory(parsedMessage.payload.chats || []);
          }
          if (typeof setIsLoadingHistory === 'function') {
            setIsLoadingHistory(false);
          }
          break;
        case 'chat_renamed':
        case 'chat_deleted':
        case 'title_updated':
          // Refresh the entire chat list from the backend if a chat is modified
          console.log(`App.jsx: Received '${parsedMessage.type}', refreshing chat list.`);
          if (typeof fetchChatHistory === 'function') {
            fetchChatHistory(user.id);
          }
          break;
        case 'rename_chat_error':
        case 'delete_chat_error':
          // Log errors related to chat operations and refresh history
          console.error(`App.jsx: Error with ${parsedMessage.type}:`, parsedMessage.payload?.error);
          if (typeof fetchChatHistory === 'function') {
            fetchChatHistory(user.id);
          }
          break;
        default:
          // Default case for unhandled message types
          break;
      }
    };

    // Event handler for WebSocket errors
    socket.onerror = (error) => {
      console.error("App.jsx: Global WebSocket Error:", error);
      setIsConnected(false); // Update connection status
      if (typeof setIsLoadingHistory === 'function') {
        setIsLoadingHistory(false);
      }
    };
    
    // Event handler for WebSocket closure
    socket.onclose = () => {
      console.log("App.jsx: Global WebSocket Disconnected.");
      setIsConnected(false); // Update connection status
      ws.current = null; // Clear ref to allow re-creation on next effect run
    };
    
    // Cleanup function: runs when component unmounts or dependencies change
    return () => {
        if (socket) {
            console.log("App.jsx: Detaching WebSocket handlers during cleanup.");
            // Remove event listeners to prevent memory leaks and unexpected behavior
            socket.onopen = null;
            socket.onmessage = null;
            socket.onerror = null;
            // The onclose handler will take care of ws.current = null;
        }
    };
    
  // Dependencies array: effect re-runs when `user` object or any of the `useChatHistory` functions change.
  }, [user, fetchChatHistory, setChatHistory, setIsLoadingHistory]);


  // Define available AI models and selected model state
  const availableModels = [{ id: "default-model", name: "Default Model (App)" }, { id: "Zephyrine Unified fragmented Model Interface", name: "Zephyrine Unified Model" }];
  const [selectedModel, setSelectedModel] = useState(availableModels[0]?.id || "default-model");

  // Callback to toggle sidebar visibility (passed to Sidebar and Header)
  const toggleSidebar = useCallback(() => {
    setIsSidebarOpen((prev) => !prev);
  }, []);

  // Callback to handle creating a new chat session (passed to Sidebar)
  const handleNewChat = useCallback(() => {
    navigate(`/chat/new`); // Navigate to a new chat path
  }, [navigate]);

  // Callback to toggle the Voice Assistant Overlay visibility (passed to Sidebar)
  const toggleVoiceAssistant = useCallback(() => {
    setIsVoiceAssistantVisible(prev => !prev);
  }, []);

  /**
   * Callback to handle messages received from the Voice Assistant Overlay.
   * This function closes the overlay and can be extended to send the message to ChatPage.
   */
  const handleSendMessageFromVoiceAssistant = useCallback((messageContent) => {
    console.log("App.jsx: Message from Voice Assistant:", messageContent);
    setIsVoiceAssistantVisible(false); // Close the overlay after sending/processing the message
  }, []); 


  // Display loading screen while authentication is in progress
  if (authLoading) {
    return (
      <div className="app-loading-screen">
        <div className="loader"></div>
        <p>Loading user session...</p>
      </div>
    );
  }

  // If no user is authenticated, render the authentication component
  if (!user) {
    return <Auth />;
  }

  // Main application layout once authenticated
  return (
    <div id="content"> {/* Main container for the entire application content */}
      
      {/* NEW: Themed background container managed by the useThemedBackground hook */}
      <div ref={backgroundRef} className="background-container">
        {/* Render clouds only when the theme is 'light' */}
        {theme === 'light' && (
          <div className="clouds">
            <div className="cloud c1"></div>
            <div className="cloud c2"></div>
            <div className="cloud c3"></div>
          </div>
        )}

        {/* NEW: Conditionally render the canvas here, and pass its ref */}
        {theme === 'dark' && (
          <canvas ref={starsCanvasRef} className="stars-canvas"></canvas>
        )}
      </div>

      {/* Overlay that appears when sidebar is open on mobile, closes sidebar on click */}
      <div className={`sidebar-overlay ${!isSidebarOpen ? 'active' : ''}`} onClick={toggleSidebar}></div>
      
      {/* Voice Assistant Overlay: Rendered conditionally based on its visibility state */}
      <VoiceAssistantOverlay
        isVisible={isVoiceAssistantVisible} // Prop: Controls if the overlay is displayed
        toggleVoiceAssistant={toggleVoiceAssistant} // Prop: Function to toggle overlay visibility
        onSendMessage={handleSendMessageFromVoiceAssistant} // Prop: Callback for when a message is sent from the voice assistant
      />

      {/* Main application content area. Blurred when voice assistant is active. */}
      <div id="main-app-content" style={{ filter: isVoiceAssistantVisible ? 'blur(4px)' : 'none' }}>
        <div className="logo"><img src="/img/ProjectZephyrine023LogoRenewal.png" alt="Project Zephyrine Logo" className="project-logo"/></div>
        
        <div id="main"> {/* Main layout section for sidebar and chat/page area */}
          <SystemOverlay systemInfo={systemInfo} /> {/* Displays system information */}
          
          <div className={`main-content-area ${!isSidebarOpen ? "main-content-area--sidebar-collapsed" : ""}`}>
            {/* Sidebar component */}
            <SideBar
              isCollapsed={!isSidebarOpen} // Prop: Controls sidebar's manual collapsed state
              toggleSidebar={toggleSidebar} // Prop: Function to toggle sidebar's manual state
              user={user} // Prop: Current authenticated user
              onNewChat={handleNewChat} // Prop: Callback for creating a new chat
              chats={chatHistory} // Prop: List of chat history to display
              isLoadingHistory={isLoadingHistory} // Prop: Loading state for chat history
              onRenameChat={handleRenameChat} // Prop: Callback for renaming a chat (from useChatHistory)
              onDeleteChat={handleDeleteChat} // Prop: Callback for deleting a chat (from useChatHistory)
              availableModels={availableModels} // Prop: List of available AI models
              selectedModel={selectedModel} // Prop: Currently selected AI model
              onModelChange={setSelectedModel} // Prop: Callback for changing the selected AI model
              onActivateVoiceMode={toggleVoiceAssistant} // Prop: Callback to activate (open) the voice assistant overlay
              isConnected={isConnected} // Prop: WebSocket connection status
            />
            
            {/* Area where different application routes are rendered */}
            <div className="chat-area-wrapper">
              <Suspense fallback={<div className="loading-screen">Loading Page...</div>}>
                <Routes>
                  <Route path="/" element={<RedirectToNewChat />} /> {/* Redirects to new chat */}
                  <Route
                    path="/chat/:chatId"
                    element={
                      <ChatPage
                        systemInfo={systemInfo}
                        user={user}
                        selectedModel={selectedModel}
                        getWsInstance={getWsInstance} // Pass WebSocket getter to ChatPage
                        isConnected={isConnected} // Pass WebSocket connection status
                        updateSidebarHistory={setChatHistory} // Pass setter for sidebar history updates
                        triggerSidebarRefresh={() => fetchChatHistory(user.id)} // Trigger full history refresh
                        onVoiceMessageSend={handleSendMessageFromVoiceAssistant} // Pass voice message handler to ChatPage
                      />
                    }
                  />
                  <Route path="/knowledge-tuning" element={<KnowledgeTuningPage />} />
                  <Route path="/images" element={<ImageGenerationPage />} />
                  <Route path="*" element={<Navigate to="/" replace />} /> {/* Catch-all route */}
                </Routes>
              </Suspense>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// The main App component which sets up the AuthProvider
// It is wrapped by ThemeProvider and Router in src/main.jsx
const App = () => (
  <AuthProvider>
    <AppContent /> 
  </AuthProvider>
);

export default App;