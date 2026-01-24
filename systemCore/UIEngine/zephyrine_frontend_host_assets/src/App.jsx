// externalAnalyzer_GUI/frontend-face-zephyrine/src/App.jsx
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
import VoiceAssistantPage from "./components/VoiceAssistantPage";
import RedirectToNewChat from "./components/RedirectToNewChat";
import SystemInfo from "./components/SystemInfo";
import SplashScreen from "./components/SplashScreen";

// NEW: Import PreSplashScreen
import PreSplashScreen from "./components/PreSplashScreen";

import ShuttleDisplay from "./components/ShuttleDisplay"; 
import SettingsModal from "./components/SettingsModal";
import ThemedBackground from "./components/ThemedBackground";

// Hook Imports
import { useSystemInfo } from "./hooks/useSystemInfo";
import { useChatHistory } from "./hooks/useChatHistory";
import { useThemedBackground } from './hooks/useThemedBackground';


import { useStarBackground } from './hooks/useStarBackground';
import StarParticle from './components/StarParticle';
import { useWingModeTransition } from './hooks/useWingModeTransition'; 


// Stylesheet Imports
import "./styles/App.css";
import "./styles/index.css";
import "./styles/Header.css";
import "./styles/ChatInterface.css";
import "./styles/SystemInfo.css";
import 'katex/dist/katex.min.css';
import "./styles/components/_splashScreen.css";
// NEW: Import pre-splash screen CSS
import "./styles/components/_preSplashScreen.css";
import "./styles/components/_wingModePage.css"; 



//const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL || "ws://localhost:3001";
import { FrontendBackendRecieve } from './config'; // Import the helper
const backendHttpUrl = window.FrontendBackendRecieve || "http://localhost:3001";
//const WEBSOCKET_URL = backendHttpUrl.replace(/^http/, 'ws');
const WEBSOCKET_URL = backendHttpUrl.replace(/^http/, 'ws') + "/zepzepadaui";
const AppContent = () => {
  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();

  const ws = useRef(null);

  //const { theme } = useTheme();
  const { theme, toggleTheme } = useTheme();
  const [forcedTheme, setForcedTheme] = useState(theme);
  useEffect(() => {
      if (forcedTheme) {
          document.documentElement.className = forcedTheme;
      } else {
          document.documentElement.className = theme;
      }
  }, [theme, forcedTheme]);
  //const backgroundRef = useRef(null);

  //useThemedBackground(backgroundRef);

  //const stars = useStarBackground();

  // NEW: State for pre-splash screen visibility
  const [showPreSplashScreen, setShowPreSplashScreen] = useState(true);
  const [showSplashScreen, setShowSplashScreen] = useState(false); // Initial state set to false
  const [appReady, setAppReady] = useState(false);
  const [isSettingsModalVisible, setIsSettingsModalVisible] = useState(false);
  

  const isWingModeActive = useWingModeTransition(); 


  // NEW: Callback for when pre-splash screen determines system is ready
  const handlePrimedAndReady = useCallback(() => {
    setShowPreSplashScreen(false); // Hide pre-splash
    setShowSplashScreen(true); // Show main splash screen
    // The main splash screen's useEffect will then handle its own fade-out
  }, []);

  // Callback for when the main splash screen's fade-out animation completes
  const handleSplashScreenFadeOutComplete = useCallback(() => {
    setShowSplashScreen(false);
    setAppReady(true); // Make main app content interactive
  }, []);


  const handleSettingsClick = useCallback(() => {
    setIsSettingsModalVisible(true);
  }, []);

  const handleCloseSettings = useCallback(() => {
    setIsSettingsModalVisible(false);
  }, []);

  const handleApplySettings = async (newConfig) => {
    const response = await fetch('/ZephyCortexConfig', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newConfig),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || 'Failed to apply settings');
    }
    return await response.json();
  };

  const handleForceThemeChange = useCallback((newTheme) => {
    setForcedTheme(newTheme);
  }, []);


  // useEffect to manage main splash screen animation sequence
  useEffect(() => {
    let logoFadeOutTimer;
    let overlayFadeOutTimer;

    // Only run this logic if the main splash screen is currently active
    if (showSplashScreen) {
      const overlayElement = document.querySelector('.splash-screen-overlay');
      if (overlayElement) {
        overlayElement.classList.add('visible'); // Ensure it becomes visible to start animations
      }

      // Phase 2: After logo fade-in (CSS handled by animation-delay), start fade-out
      logoFadeOutTimer = setTimeout(() => {
        const logoElement = document.querySelector('.splash-screen-logo');
        if (logoElement) {
          logoElement.style.animation = 'splash-logo-fade-out 1s ease-in forwards';
        }
      }, 2500); 

      // Phase 3: After logo fades out (1s duration), fade out the overlay
      overlayFadeOutTimer = setTimeout(() => {
        const overlayElement = document.querySelector('.splash-screen-overlay');
        if (overlayElement) {
          overlayElement.classList.remove('visible'); // Triggers fade-out via CSS
        }
      }, 16000); // Start overlay fade-out (TimeBudget)
      // Splash Screen timing total budget (If you add something on thhe Splash Screen and need additional time, don't forget to add this too keyword: TotalTimefraemBudgetSplash)

      // Phase 4: This will now be handled by onTransitionEnd in SplashScreen.jsx -> handleSplashScreenFadeOutComplete
    }

    return () => {
      clearTimeout(logoFadeOutTimer);
      clearTimeout(overlayFadeOutTimer);
    };
  }, [showSplashScreen]); // Depend on showSplashScreen


  const getWsInstance = useCallback(() => ws.current, []); 

  const { systemInfo } = useSystemInfo();
  
  const {
    chatHistory,
    isLoadingHistory,
    fetchChatHistory,
    setChatHistory,
    setIsLoadingHistory,
    handleRenameChat, 
    handleDeleteChat
  } = useChatHistory(getWsInstance);

  // MODIFIED: Set default state to false (Collapsed)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  //const [isVoiceAssistantVisible, setIsVoiceAssistantVisible] = useState(false); 
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!user?.id) {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        console.log("App.jsx: No user, closing global WebSocket.");
        ws.current.close();
      }
      ws.current = null;
      setIsConnected(false);
      return;
    }

    if (!ws.current || ws.current.readyState === WebSocket.CLOSED) {
      console.log("App.jsx: Connecting global WebSocket...");
      ws.current = new WebSocket(WEBSOCKET_URL);
      setIsConnected(false);
    }

    console.log("App.jsx: Attaching WebSocket event handlers.");

    const socket = ws.current;

    socket.onopen = () => {
      console.log("App.jsx: Global WebSocket Connected.");
      setIsConnected(true);
      if (typeof fetchChatHistory === 'function') {
        fetchChatHistory(user.id);
      }
    };

    socket.onmessage = (event) => {
      const parsedMessage = JSON.parse(event.data);
      console.log("App.jsx: Global WS Message Received:", parsedMessage.type);

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
        default:
          break;
      }
    };

    socket.onerror = (error) => {
      console.error("App.jsx: Global WebSocket Error:", error);
      setIsConnected(false);
      if (typeof setIsLoadingHistory === 'function') {
        setIsLoadingHistory(false);
      }
    };
    
    socket.onclose = () => {
      console.log("App.jsx: Global WebSocket Disconnected.");
      setIsConnected(false);
      ws.current = null;
    };
    
    return () => {
        if (socket) {
            console.log("App.jsx: Detaching WebSocket handlers during cleanup.");
            socket.onopen = null;
            socket.onmessage = null;
            socket.onerror = null;
        }
    };
    
  }, [user, fetchChatHistory, setChatHistory, setIsLoadingHistory]);


  const availableModels = [{ id: "+11111111111", name: "Default Model (App)" }, { id: "Zephyrine Unified fragmented Model Interface", name: "Snowball Enaga Model" }];
  const [selectedModel, setSelectedModel] = useState(availableModels[0]?.id || "+11111111111");

  const toggleSidebar = useCallback(() => {
    setIsSidebarOpen((prev) => !prev);
  }, []);

  const handleNewChat = useCallback(() => {
    navigate(`/`);
  }, [navigate]);

  //const toggleVoiceAssistant = useCallback(() => {
  //  setIsVoiceAssistantVisible(prev => !prev);
  //}, []);


  if (authLoading) {
    return (
      <div className="app-loading-screen">
        <div className="loader"></div>
        <p>Loading user session...</p>
      </div>
    );
  }

  // NEW: Render PreSplashScreen first
  if (showPreSplashScreen) {
    return <PreSplashScreen onPrimedAndReady={handlePrimedAndReady} />;
  }

  // If Wing Mode is active, render the WingModePage and nothing else.
  // This will overlay everything.
  if (isWingModeActive) {
  return <ShuttleDisplay />;
}

  if (!user) {
    return <Auth />;
  }

  return (
    <div id="content">
      {/* Main SplashScreen component is now visible only when triggered */}
      {showSplashScreen && <SplashScreen isVisible={showSplashScreen} onFadeOutComplete={handleSplashScreenFadeOutComplete} />}

      <ThemedBackground />

      {/* MODIFIED: Overlay should be active when sidebar IS open, not when it is NOT open */}
      <div className={`sidebar-overlay ${isSidebarOpen ? 'active' : ''}`} onClick={toggleSidebar}></div>
    

      <div id="main-app-content" style={{ 
          opacity: appReady ? 1 : 0,
          transition: 'opacity 0.5s ease-out',
          pointerEvents: appReady ? 'auto' : 'none',
      }}>
        
        <div id="main">

          
          <div className={`main-content-area ${!isSidebarOpen ? "main-content-area--sidebar-collapsed" : ""}`}>
          <SideBar
            isCollapsed={!isSidebarOpen}
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
            isConnected={isConnected}
            onSettingsClick={handleSettingsClick}
          />
            
            <SettingsModal
                isVisible={isSettingsModalVisible}
                onClose={handleCloseSettings}
                onApply={handleApplySettings}
                // A "critical mission" (like AI generation) state isn't available in App.jsx.
                // Passing `false` satisfies the requirement and is the correct default behavior.
                isCriticalMission={false} 
                onForceThemeChange={handleForceThemeChange}
                currentTheme={forcedTheme}
            />

            <div className="chat-area-wrapper">
              <Suspense fallback={<div className="loading-screen">Loading Page...</div>}>
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
                        triggerSidebarRefresh={() => fetchChatHistory(user.id)}
                      />
                    }
                  />
                  <Route path="/knowledge-tuning" element={<KnowledgeTuningPage />} />
                  <Route path="/images" element={<ImageGenerationPage />} />
                  <Route path="/voice-assistant" element={<VoiceAssistantPage />} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </Suspense>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const App = () => (
  <AuthProvider>
    <AppContent /> 
  </AuthProvider>
);

export default App;