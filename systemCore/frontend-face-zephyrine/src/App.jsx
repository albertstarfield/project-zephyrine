// ExternalAnalyzer/frontend-face-zephyrine/src/App.jsx
import { useState, useEffect, useCallback } from 'react';
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

import { useStarBackground } from './hooks/useStarBackground';
import { useSystemInfo } from './hooks/useSystemInfo';
import { useChatHistory } from './hooks/useChatHistory';

function App() {
  console.log("<<<<< App.jsx Render/Re-render - TOP >>>>>");

  const { user } = useAuth();
  const navigate = useNavigate();

  const systemInfo = useSystemInfo();
  const stars = useStarBackground();

  const {
    chatHistory,
    isLoadingHistory,
    fetchChatHistory, // This function should be implemented in useChatHistory to use WebSocket
    setChatHistory,   // This is the state updater from useChatHistory
    handleRenameChat,
    handleDeleteChat
  } = useChatHistory(); // Assuming useChatHistory does not take ws, App will manage ws calls for history list if needed

  // Diagnostic log for functions from useChatHistory
  useEffect(() => {
    console.log("App.jsx - DIAGNOSTIC: typeof setChatHistory from useChatHistory:", typeof setChatHistory);
    console.log("App.jsx - DIAGNOSTIC: typeof fetchChatHistory from useChatHistory:", typeof fetchChatHistory);
  }, [setChatHistory, fetchChatHistory]);


  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(true);
  
  const availableModels = [
    {id: "default-model", name: "Default Model (App)"},
    {id: "Zephyrine Unified fragmented Model Interface", name: "Zephyrine Unified Model"}
  ];
  const [selectedModel, setSelectedModel] = useState(availableModels[0]?.id || "default-model"); 

  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed);
  };

  useEffect(() => {
    const handleResize = () => {};
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const handleNewChat = () => {
    const newChatId = uuidv4();
    console.log("App.jsx: Creating new chat with ID:", newChatId);
    navigate(`/chat/${newChatId}`);
    // Sidebar will update when ChatPage's WS connection fetches the list or a title_updated event is handled
  };
  
  // Memoized callback for triggerSidebarRefresh
  // This function's purpose is to tell useChatHistory hook to initiate a fetch.
  // The hook itself would then need to make the WS call.
  const triggerSidebarRefreshCallback = useCallback(() => {
    console.log("App.jsx: triggerSidebarRefreshCallback called. User ID:", user?.id);
    if (user?.id && typeof fetchChatHistory === 'function') {
      fetchChatHistory(user.id); // This signals the hook to fetch
    } else {
      if (typeof fetchChatHistory !== 'function') {
        console.warn("App.jsx: fetchChatHistory from useChatHistory is not a function in triggerSidebarRefreshCallback!");
      }
      if (!user?.id) {
        console.warn("App.jsx: triggerSidebarRefreshCallback called but no user.id available.");
      }
    }
  }, [user, fetchChatHistory]);

  // This useEffect is for an initial fetch if App were managing a central WebSocket.
  // Since ChatPage initiates its own WS and fetches the list, this might be redundant for *initial* load
  // but good for when `user` changes if the app stays mounted.
  useEffect(() => {
    if (user?.id) {
      console.log("App.jsx: User is available. Initial chat list fetch is handled by ChatPage on its WebSocket connection open. This effect could trigger a redundant refresh if not careful.");
      // triggerSidebarRefreshCallback(); // Potentially call here if ChatPage doesn't fetch on its own
    }
  }, [user, triggerSidebarRefreshCallback]);

  console.log("App.jsx: Props being prepared for ChatPage. typeof setChatHistory:", typeof setChatHistory);

  return (
    <div id="content">
      <div className={`sidebar-overlay ${!isSidebarCollapsed ? 'active' : ''}`} onClick={toggleSidebar}></div>
      <div id="sky">
      {stars && stars.map(star => (
        star && star.style ? (
          <div
            key={star.id || Math.random()}
            className="star"
            style={{
              left: star.left,
              top: star.top,
              width: star.width,
              height: star.height,
              animationDelay: star.style.animationDelay, 
            }}
          />
        ) : null 
      ))}
      </div>
      <div className="logo">
        <img
          src="/img/ProjectZephy023LogoRenewal.png"
          alt="Project Zephyrine Logo"
          className="project-logo"
        />
      </div>
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
            onRenameChat={handleRenameChat} // Assumes useChatHistory provides a working version
            onDeleteChat={handleDeleteChat}   // Assumes useChatHistory provides a working version
            availableModels={availableModels} 
            selectedModel={selectedModel}     
            onModelChange={setSelectedModel}
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
                    refreshHistory={fetchChatHistory} // Keep if ChatPage uses it directly for some reason
                    selectedModel={selectedModel} 
                    // --- Crucial Props for Sidebar History ---
                    updateSidebarHistory={setChatHistory} // Pass the actual setChatHistory function
                    triggerSidebarRefresh={triggerSidebarRefreshCallback} // Pass the memoized callback
                    // --- End Crucial Props ---
                  />
                }
              />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;