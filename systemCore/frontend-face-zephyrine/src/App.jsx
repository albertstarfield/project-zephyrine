import { useState, useEffect, useCallback } from 'react'; // Keep necessary imports
import { Routes, Route, useNavigate, Navigate } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';
import { useAuth } from './contexts/AuthContext';
// Removed: import Auth from './components/Auth'; // Remove if Auth component is no longer used
import './styles/App.css';
import SideBar from './components/SideBar';
import SystemOverlay from './components/SystemOverlay';
import ChatPage from './components/ChatPage';
import './styles/ChatInterface.css';
import './styles/utils/_overlay.css';
import RedirectToNewChat from './components/RedirectToNewChat';

// Import custom hooks
import { useStarBackground } from './hooks/useStarBackground';
import { useSystemInfo } from './hooks/useSystemInfo';
import { useChatHistory } from './hooks/useChatHistory'; // Keep this hook

// Main App component handles layout, routing, and authentication check
function App() {
  // _s(); // REMOVE THIS LINE (was line 32)

  // Since AuthContext now provides a dummy session immediately, loading should always be false
  // and session/user should always exist.
  const { session, user } = useAuth(); // Get dummy auth state
  const navigate = useNavigate();

  // Use custom hooks for state management
  const systemInfo = useSystemInfo();
  const stars = useStarBackground();

  // useChatHistory hook (Needs adaptation for WebSocket fetching/updates)
  const {
    chatHistory,
    isLoadingHistory, // Get loading state from the hook
    fetchChatHistory, // Function to trigger a fetch (needs WS implementation)
    handleRenameChat, // Needs WS implementation for backend call
    handleDeleteChat  // Needs WS implementation for backend call
  } = useChatHistory();

  // Sidebar state
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  // Toggle sidebar function
  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed);
  };

  // Effect for handling resize
  useEffect(() => {
    const handleResize = () => {
      // Adjust behavior on resize if needed
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []); // Removed dependency on isSidebarCollapsed if not needed inside

  // Function to handle creating a new chat
  const handleNewChat = () => {
    const newChatId = uuidv4();
    console.log("Creating new chat with ID:", newChatId);
    // TODO: Send a message to backend via WebSocket to potentially create
    // the chat entry in DB immediately? Or wait for first message?
    // Example: ws.send(JSON.stringify({ type: "create_chat", payload: { chatId: newChatId, userId: user.id } }));
    navigate(`/chat/${newChatId}`);
    // History will be updated when fetchChatHistory is called or based on WS messages
  };

  // No need for the loading check or !session check if using the dummy AuthContext
  /*
  if (loading) { // This block can likely be removed
    return <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>Loading Auth...</div>;
  }
  if (!session) { // This block can likely be removed
    // return <Auth />; // Remove the Auth component import/usage
    return <div>Please Log In (Auth Component Removed)</div>; // Placeholder if needed
  }
  */

  // Render the main app using hooks
  return (
    <div id="content">
      {/* Overlay for mobile sidebar */}
      <div className="sidebar-overlay" onClick={toggleSidebar}></div>

      {/* Star Background */}
      <div id="sky">
        {stars.map((star) => (
          <div
            key={star.id}
            className="star"
            style={{
              left: star.left, top: star.top, width: star.size, height: star.size,
              animation: `twinkling ${star.animationDuration} infinite ${star.animationDelay}`,
            }}
          />
        ))}
      </div>

      {/* Logo */}
      <div className="logo">
        <img
          src="/img/ProjectZephy023LogoRenewal.png"
          alt="Project Zephyrine Logo"
          className="project-logo"
        />
      </div>

      {/* Main Content */}
      <div id="main">
        <SystemOverlay systemInfo={systemInfo} />

        <div className={`main-content-area ${isSidebarCollapsed ? "main-content-area--sidebar-collapsed" : ""}`}>
          {/* Sidebar */}
          <SideBar
            systemInfo={systemInfo}
            isCollapsed={isSidebarCollapsed}
            toggleSidebar={toggleSidebar}
            user={user} // From dummy AuthContext
            onNewChat={handleNewChat}
            chatHistory={chatHistory} // From useChatHistory (needs proper fetching)
            isLoadingHistory={isLoadingHistory} // Pass loading state
            onRenameChat={handleRenameChat} // From useChatHistory (needs WS call)
            onDeleteChat={handleDeleteChat} // From useChatHistory (needs WS call)
            // Removed model selection props
            // Pass WebSocket related functions if needed for rename/delete
          />

          {/* Chat Area Wrapper */}
          <div className="chat-area-wrapper">
            <Routes>
              <Route path="/" element={<RedirectToNewChat />} />
              <Route
                path="/chat/:chatId"
                element={
                  <ChatPage
                    systemInfo={systemInfo}
                    user={user} // Pass user from dummy AuthContext
                    refreshHistory={fetchChatHistory} // Pass refresh trigger
                    // Pass selectedModel if needed (using a fixed value or state)
                    selectedModel={"default-model"} // Example fixed value
                    // Pass WebSocket instance or send function if needed directly
                  />
                }
              />
              <Route path="*" element={<Navigate to="/" replace />} /> {/* Catch-all redirect */}
            </Routes>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;