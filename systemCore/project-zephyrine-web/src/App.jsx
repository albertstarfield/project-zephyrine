import { useState, useEffect, useCallback } from 'react'; // Removed useRef, useParams
import {
  Routes,
  Route,
  // useParams is used in ChatPage
  useNavigate,
  Navigate,
  // useLocation is used in useChatHistory hook
} from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';
// supabase is used in useChatHistory hook
import { useAuth } from './contexts/AuthContext'; // Import useAuth
import Auth from './components/Auth'; // Import Auth component
import './styles/App.css';
import SideBar from './components/SideBar';
// ChatFeed and InputArea are now used within ChatPage
import SystemOverlay from './components/SystemOverlay';
import ChatPage from './components/ChatPage'; // Import the extracted component
import './styles/ChatInterface.css';
import './styles/utils/_overlay.css';
import RedirectToNewChat from './components/RedirectToNewChat'; // Import the extracted component

// Import custom hooks
import { useStarBackground } from './hooks/useStarBackground';
import { useSystemInfo } from './hooks/useSystemInfo';
import { useChatHistory } from './hooks/useChatHistory';
import { useModelSelection } from './hooks/useModelSelection';


// Main App component handles layout, routing, and authentication check
function App() {
  const { session, loading, user } = useAuth(); // Get auth state
  const navigate = useNavigate();

  // Use custom hooks for state management
  const systemInfo = useSystemInfo();
  const stars = useStarBackground();
  const {
    chatHistory,
    fetchChatHistory, // Keep fetchChatHistory to pass to ChatPage for refresh
    handleRenameChat,
    handleDeleteChat
  } = useChatHistory();
  const {
    availableModels,
    selectedModel,
    handleModelChange
  } = useModelSelection();

  // Sidebar state remains simple, keep it here for now
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  // Toggle sidebar function
  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed);
  };

  // Effect for handling resize and mobile overlay remains relevant to sidebar state
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > 767 && !isSidebarCollapsed) {
        // Optional: Adjust behavior on resize to desktop
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [isSidebarCollapsed]);


  // Function to handle creating a new chat (uses navigate)
  const handleNewChat = () => {
    const newChatId = uuidv4();
    navigate(`/chat/${newChatId}`);
    // History is updated when the first message is sent in the new chat via fetchChatHistory
  };

  // Show loading indicator during initial auth check
  if (loading) {
    return <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>Loading...</div>;
  }

  // If no session, show Auth component
  if (!session) {
    return <Auth />;
  }

  // If session exists, render the main app using hooks
  return (
    <div id="content">
      {/* Overlay for mobile sidebar */}
      {!isSidebarCollapsed && (
        <div className="sidebar-overlay" onClick={toggleSidebar}></div>
      )}

      {/* Star Background */}
      <div id="sky">
        {stars.map((star) => (
          <div
            key={star.id}
            className="star"
            style={{
              left: star.left,
              top: star.top,
              width: star.size,
              height: star.size,
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
        {/* System Overlay - Consider if this needs systemInfo or if it can be self-contained */}
        <SystemOverlay /* Pass systemInfo if needed: systemInfo={systemInfo} */ />

        {/* Main Layout Area */}
        <div
          className={`main-content-area ${
            isSidebarCollapsed ? "main-content-area--sidebar-collapsed" : ""
          }`}
        >
          {/* Sidebar using hook values */}
          <SideBar
            systemInfo={systemInfo} // From useSystemInfo
            isCollapsed={isSidebarCollapsed}
            toggleSidebar={toggleSidebar}
            user={user} // From useAuth
            onNewChat={handleNewChat}
            chatHistory={chatHistory} // From useChatHistory
            onRenameChat={handleRenameChat} // From useChatHistory
            onDeleteChat={handleDeleteChat} // From useChatHistory
            availableModels={availableModels} // From useModelSelection
            selectedModel={selectedModel} // From useModelSelection
            onModelChange={handleModelChange} // From useModelSelection
          />

          {/* Chat Area Wrapper */}
          <div className="chat-area-wrapper">
            <Routes>
              <Route path="/" element={<RedirectToNewChat />} />
              <Route
                path="/chat/:chatId"
                element={
                  <ChatPage
                    systemInfo={systemInfo} // Pass systemInfo from hook
                    user={user} // Pass user from useAuth
                    refreshHistory={fetchChatHistory} // Pass refresh function from useChatHistory
                    selectedModel={selectedModel} // Pass selectedModel from hook
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
