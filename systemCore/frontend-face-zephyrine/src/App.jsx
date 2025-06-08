// ExternalAnalyzer/frontend-face-zephyrine/src/App.jsx
import { useState, useEffect, useCallback, useRef } from 'react';
import { Routes, Route, useNavigate, Navigate } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';
import { useAuth } from './contexts/AuthContext';
import './styles/App.css';
import SideBar from './components/SideBar';
import SystemOverlay from './components/SystemOverlay';
import ChatPage from './components/ChatPage';
import './styles/ChatInterface.css';
import './styles/utils/_overlay.css';
import RedirectToNewChat from './components/RedirectToNewChat';
import VoiceAssistantOverlay from './components/VoiceAssistantOverlay';

import { useStarBackground } from './hooks/useStarBackground';
import { useSystemInfo } from './hooks/useSystemInfo';
import { useChatHistory } from './hooks/useChatHistory';

const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL || "ws://localhost:3001";

function App() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const ws = useRef(null); // Ref to hold the single WebSocket instance

  const systemInfo = useSystemInfo();
  const stars = useStarBackground();
  
  // A stable function to provide access to the WebSocket instance
  const getWsInstance = useCallback(() => ws.current, []); 

  const {
    chatHistory,
    isLoadingHistory,
    fetchChatHistory,
    setChatHistory,
    setIsLoadingHistory,
    updateSidebarHistory, // This function is returned by the hook
    handleRenameChat,
    handleDeleteChat
  } = useChatHistory(getWsInstance); // Pass the getter function to the hook

  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(true);
  const [isVoiceModeActive, setIsVoiceModeActive] = useState(false);
  const [isConnected, setIsConnected] = useState(false);

  // Centralized WebSocket management
  useEffect(() => {
    // 1. If there is no user, ensure the WebSocket is closed and exit.
    if (!user?.id) {
      if (ws.current) {
        console.log("App.jsx: No user, closing global WebSocket.");
        ws.current.close();
        ws.current = null;
      }
      return;
    }

    // 2. If the WebSocket doesn't exist or is closed, create a new one.
    //    This part only runs when absolutely necessary.
    if (!ws.current || ws.current.readyState === WebSocket.CLOSED) {
      console.log("App.jsx: Connecting global WebSocket...");
      ws.current = new WebSocket(WEBSOCKET_URL);
      setIsConnected(false); // Set connecting state
    }

    // 3. Always re-assign event handlers to prevent stale closures.
    //    This ensures they always use the latest version of any functions from state or props.
    console.log("App.jsx: Attaching WebSocket event handlers.");

    ws.current.onopen = () => {
      console.log("App.jsx: Global WebSocket Connected.");
      setIsConnected(true);
      // Use the fetch function from the hook to get initial history
      if (typeof fetchChatHistory === 'function') {
        fetchChatHistory(user.id);
      }
    };

    ws.current.onmessage = (event) => {
      const parsedMessage = JSON.parse(event.data);
      console.log("App.jsx: Global WS Message Received:", parsedMessage.type);

      // This handler now safely uses the latest functions
      switch (parsedMessage.type) {
        case 'chat_history_list':
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
          console.log(`App.jsx: Received '${parsedMessage.type}', refreshing chat list.`);
          if (typeof fetchChatHistory === 'function') {
            fetchChatHistory(user.id);
          }
          break;
        case 'rename_chat_error':
        case 'delete_chat_error':
          console.error(`App.jsx: Error with ${parsedMessage.type}:`, parsedMessage.payload?.error);
          if (typeof fetchChatHistory === 'function') {
            fetchChatHistory(user.id);
          }
          break;
      }
    };

    ws.current.onerror = (error) => {
      console.error("App.jsx: Global WebSocket Error:", error);
      setIsConnected(false);
      if (typeof setIsLoadingHistory === 'function') {
        setIsLoadingHistory(false);
      }
    };
    
    ws.current.onclose = () => {
      console.log("App.jsx: Global WebSocket Disconnected.");
      setIsConnected(false);
      ws.current = null; // Set to null so it can be re-created
    };
    
    // 4. The cleanup function runs when the user changes or the component unmounts.
    //    It's good practice to remove handlers to prevent memory leaks,
    //    though closing the connection on user logout is the most critical part.
    return () => {
        if (ws.current) {
            console.log("App.jsx: Detaching WebSocket handlers during cleanup.");
            ws.current.onopen = null;
            ws.current.onmessage = null;
            ws.current.onerror = null;
            // The onclose logic will handle the rest
        }
    };
    
  // The dependency array correctly triggers this effect when the user or a function changes.
  }, [user, fetchChatHistory, setChatHistory, setIsLoadingHistory]);


  const availableModels = [{ id: "default-model", name: "Default Model (App)" }, { id: "Zephyrine Unified fragmented Model Interface", name: "Zephyrine Unified Model" }];
  const [selectedModel, setSelectedModel] = useState(availableModels[0]?.id || "default-model");
  const toggleSidebar = () => setIsSidebarCollapsed(!isSidebarCollapsed);
  const handleNewChat = () => { navigate(`/chat/${uuidv4()}`); };
  const activateVoiceMode = () => setIsVoiceModeActive(true);
  const deactivateVoiceMode = () => setIsVoiceModeActive(false);

  return (
    <div id="content">
      <div className={`sidebar-overlay ${!isSidebarCollapsed ? 'active' : ''}`} onClick={toggleSidebar}></div>
      {isVoiceModeActive && <VoiceAssistantOverlay onExit={deactivateVoiceMode} />}
      <div id="main-app-content" style={{ filter: isVoiceModeActive ? 'blur(4px)' : 'none' }}>
        <div id="sky">
          {stars && stars.map(star => star && star.style ? <div key={star.id || Math.random()} className="star" style={{ ...star, animationDelay: star.style.animationDelay }} /> : null)}
        </div>
        <div className="logo"><img src="/img/ProjectZephy023LogoRenewal.png" alt="Project Zephyrine Logo" className="project-logo"/></div>
        <div id="main">
          <SystemOverlay systemInfo={systemInfo} />
          <div className={`main-content-area ${isSidebarCollapsed ? "main-content-area--sidebar-collapsed" : ""}`}>
          <SideBar
            isCollapsed={isSidebarCollapsed}
            toggleSidebar={toggleSidebar}
            user={user} 
            onNewChat={handleNewChat}
            chats={chatHistory} 
            isLoadingHistory={isLoadingHistory}
            onRenameChat={handleRenameChat} 
            onDeleteChat={handleDeleteChat}   
            availableModels={availableModels} 
            selectedModel={selectedModel}     
            onModelChange={setSelectedModel}
            onActivateVoiceMode={activateVoiceMode}
            isConnected={isConnected} // <<<<< ADD THIS LINE
          />
            <div className="chat-area-wrapper">
              <Routes>
                <Route path="/" element={<RedirectToNewChat />} />
                <Route
                  path="/chat/:chatId"
                  element={
                    <ChatPage
                      systemInfo={systemInfo}
                      user={user} 
                      selectedModel={selectedModel} 
                      getWsInstance={getWsInstance} 
                      isConnected={isConnected} 
                      updateSidebarHistory={setChatHistory} 
                      triggerSidebarRefresh={() => fetchChatHistory(user.id)} // Add this line
                    />
                  }
                />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;