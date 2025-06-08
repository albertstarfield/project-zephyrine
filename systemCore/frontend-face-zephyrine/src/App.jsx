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
  const ws = useRef(null); // Central WebSocket instance ref

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
    handleRenameChat,
    handleDeleteChat
  } = useChatHistory(getWsInstance); // Pass the getter function to the hook

  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(true);
  const [isVoiceModeActive, setIsVoiceModeActive] = useState(false);
  const [isConnected, setIsConnected] = useState(false);

  const availableModels = [
    { id: "default-model", name: "Default Model (App)" },
    { id: "Zephyrine Unified fragmented Model Interface", name: "Zephyrine Unified Model" }
  ];
  const [selectedModel, setSelectedModel] = useState(availableModels[0]?.id || "default-model");

  // Centralized WebSocket management
  useEffect(() => {
    if (!user?.id) {
      ws.current?.close();
      return;
    }

    if (!ws.current || ws.current.readyState === WebSocket.CLOSED) {
      console.log("App.jsx: Connecting global WebSocket...");
      ws.current = new WebSocket(WEBSOCKET_URL);
      setIsConnected(false);

      ws.current.onopen = () => {
        console.log("App.jsx: Global WebSocket Connected.");
        setIsConnected(true);
        // Initial history fetch is now handled here, by the hook, using the new WS instance
        fetchChatHistory(user.id);
      };

      ws.current.onmessage = (event) => {
        const parsedMessage = JSON.parse(event.data);
        console.log("App.jsx: Global WS Message Received:", parsedMessage.type);

        // This handler now processes GLOBAL messages that affect the whole app, like the chat list
        switch (parsedMessage.type) {
          case 'chat_history_list':
            setChatHistory(parsedMessage.payload.chats || []);
            setIsLoadingHistory(false);
            break;
          case 'chat_renamed':
          case 'chat_deleted':
          case 'title_updated':
            console.log(`App.jsx: Received '${parsedMessage.type}', refreshing chat list.`);
            fetchChatHistory(user.id);
            break;
          case 'rename_chat_error':
          case 'delete_chat_error':
            console.error(`App.jsx: Error with ${parsedMessage.type}:`, parsedMessage.payload?.error);
            fetchChatHistory(user.id); // Re-fetch to ensure UI consistency
            break;
        }
      };

      ws.current.onerror = (error) => {
        console.error("App.jsx: Global WebSocket Error:", error);
        setIsConnected(false);
        setIsLoadingHistory(false);
      };

      ws.current.onclose = () => {
        console.log("App.jsx: Global WebSocket Disconnected.");
        setIsConnected(false);
        ws.current = null;
        setIsLoadingHistory(false);
      };
    }
    
    // The cleanup should be handled when the App unmounts or the user logs out
    return () => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        console.log("App.jsx: Closing WebSocket connection on cleanup.");
        ws.current.close();
      }
    };
  }, [user, fetchChatHistory, setChatHistory, setIsLoadingHistory]);

  const toggleSidebar = () => setIsSidebarCollapsed(!isSidebarCollapsed);

  const handleNewChat = () => {
    const newChatId = uuidv4();
    console.log("App.jsx: Creating new chat with ID:", newChatId);
    navigate(`/chat/${newChatId}`);
  };

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
                      // Pass the WebSocket getter function to ChatPage
                      getWsInstance={getWsInstance}
                      isConnected={isConnected} // Pass connection status
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