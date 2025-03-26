import { useState, useEffect, useRef } from "react";
import {
  Routes,
  Route,
  useParams,
  useNavigate,
  Navigate,
  useLocation, // Import useLocation
} from "react-router-dom";
import { v4 as uuidv4 } from "uuid";
import "./styles/App.css";
import SideBar from "./components/SideBar";
import ChatFeed from "./components/ChatFeed";
import InputArea from "./components/InputArea";
import SystemOverlay from "./components/SystemOverlay";
import "./styles/ChatInterface.css"; // Import chat interface styles
import "./styles/utils/_overlay.css"; // Import overlay styles (will create/modify later)

// Component to handle individual chat sessions
function ChatPage({ systemInfo }) {
  const { chatId } = useParams(); // Get chatId from URL
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [showPlaceholder, setShowPlaceholder] = useState(true);
  const bottomRef = useRef(null);

  // Reset chat state when chatId changes (optional, depends on desired behavior)
  useEffect(() => {
    setMessages([]);
    setInputValue("");
    setIsGenerating(false);
    setShowPlaceholder(true);
    // Here you might fetch chat history based on chatId if persistence is implemented
    console.log(`Entered chat session: ${chatId}`);
  }, [chatId]);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleSendMessage = (text) => {
    if (!text.trim()) return;

    if (showPlaceholder) {
      setShowPlaceholder(false);
    }

    const userMessage = {
      id: Date.now(),
      sender: "user",
      content: text,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsGenerating(true);

    const getResponse = (input) => {
      const lowerInput = input.toLowerCase();
      if (lowerInput.includes("hello") || lowerInput.includes("hi")) {
        return "Greetings! How may I assist you today?";
      } else if (lowerInput.includes("help")) {
        return "I'd be delighted to help. Could you please provide more details about what you need assistance with?";
      } else if (
        lowerInput.includes("code") ||
        lowerInput.includes("programming")
      ) {
        return "Here's a simple example of a JavaScript function:\n\n```javascript\nfunction greet(name) {\n  return `Hello, ${name}!`;\n}\n\nconsole.log(greet('World'));\n// Output: Hello, World!\n```\n\nIs there a specific programming concept you'd like me to explain?";
      } else if (lowerInput.includes("list") || lowerInput.includes("bullet")) {
        return "Here's a list of items:\n\n* First item\n* Second item\n* Third item\n\nIs this the kind of list you were looking for?";
      } else {
        // Include chatId in the response for demonstration
        return `Processing message for chat ${chatId}: I understand your message. How can I assist you further with this topic?`;
      }
    };

    setTimeout(() => {
      const assistantMessage = {
        id: Date.now() + 1,
        sender: "assistant",
        content: getResponse(text),
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setIsGenerating(false);
    }, 1500);
  };

  const handleStopGeneration = () => {
    setIsGenerating(false);
  };

  const handleExampleClick = (text) => {
    setInputValue(text);
  };

  return (
    // Use a fragment <> to return multiple elements from ChatPage
    <>
      {/* Model Selector Placeholder - Placed at the top of the chat area */}
      {/* Only show selector if not in welcome state */}
      {!showPlaceholder && (
        <div className="chat-model-selector">
          {/* This would dynamically show the selected model */}
          <span>GPT-4o ▼</span>
          {/* Add dropdown logic later */}
        </div>
      )}

      {/* The main feed and input area */}
      <div id="feed" className={showPlaceholder ? "welcome-screen" : ""}>
        <ChatFeed
          messages={messages}
          showPlaceholder={showPlaceholder}
          isGenerating={isGenerating}
          onExampleClick={handleExampleClick}
          bottomRef={bottomRef}
          assistantName={systemInfo.assistantName}
        />
        {/* Ensure only one InputArea is rendered here */}
        <InputArea
          value={inputValue}
          onChange={setInputValue}
          onSend={handleSendMessage}
          onStop={handleStopGeneration}
          isGenerating={isGenerating}
        />
      </div>
    </> // Close the fragment
  );
}

// Component to handle redirection from root
function RedirectToNewChat() {
  const navigate = useNavigate();
  useEffect(() => {
    // Redirect to a new chat session when accessing the root path
    navigate(`/chat/${uuidv4()}`, { replace: true });
  }, [navigate]);
  return null; // Render nothing while redirecting
}

// Main App component handles layout and routing
function App() {
  const [systemInfo, setSystemInfo] = useState({
    username: "",
    assistantName: "Adelaide Zephyrine Charlotte",
    cpuUsage: 0,
    cpuFree: 0,
    cpuCount: 0,
    threadsUtilized: 0,
    freeMem: 0,
    totalMem: 0,
    os: "",
  });
  const [stars, setStars] = useState([]);
  const location = useLocation(); // Get current location
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false); // State for sidebar

  // Toggle sidebar function
  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed);
  };

  // Close sidebar if clicking outside on mobile (using overlay)
  // Also reset sidebar state if window resizes from mobile to desktop while open
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > 767 && !isSidebarCollapsed) {
        // If resizing to desktop and sidebar was open (mobile style), collapse it by default? Or keep open? Let's keep it open for now.
        // setIsSidebarCollapsed(true); // Optional: collapse on resize to desktop
      } else if (window.innerWidth <= 767 && !isSidebarCollapsed) {
        // Ensure it stays open if it was open on mobile resize
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [isSidebarCollapsed]);


  // Create stars for the background
  useEffect(() => {
    const createStars = () => {
      const newStars = [];
      const starCount = 150;
      for (let i = 0; i < starCount; i++) {
        newStars.push({
          id: i,
          left: `${Math.random() * 100}%`,
          top: `${Math.random() * 100}%`,
          size: `${Math.random() * 2 + 1}px`,
          animationDuration: `${Math.random() * 3 + 2}s`,
          animationDelay: `${Math.random() * 2}s`,
        });
      }
      setStars(newStars);
    };
    createStars();
  }, []);

  // Simulate getting system info
  useEffect(() => {
    setSystemInfo((prev) => ({
      ...prev,
      username: "User",
      cpuCount: navigator.hardwareConcurrency || 4,
      os: navigator.platform,
      totalMem: 8, // Simulated 8GB
    }));

    const interval = setInterval(() => {
      setSystemInfo((prev) => ({
        ...prev,
        cpuUsage: Math.random() * 0.5,
        cpuFree: 1 - (prev.cpuUsage || 0), // Make cpuFree dependent on cpuUsage
        freeMem: Math.random() * 4 + 2, // Between 2-6GB free
        threadsUtilized: Math.floor(
          Math.random() * (navigator.hardwareConcurrency || 4)
        ),
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Function to handle creating a new chat (can be passed to SideBar)
  const handleNewChat = () => {
    const navigate = useNavigate(); // Need to get navigate here or pass it down
    // navigate(`/chat/${uuidv4()}`); // This was defined but not used, removing for now
  };


  return (
    <div id="content">
      {/* Overlay for mobile sidebar */}
      {!isSidebarCollapsed && <div className="sidebar-overlay" onClick={toggleSidebar}></div>}

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

      <div className="logo">
        <img
          src="/img/ProjectZephy023LogoRenewal.png"
          alt="Project Zephyrine Logo"
          className="project-logo"
        />
      </div>

      <div id="main">
        <SystemOverlay />
        {/* Container for the two-column layout */}
        {/* Add class based on sidebar state */}
        <div className={`main-content-area ${isSidebarCollapsed ? "main-content-area--sidebar-collapsed" : ""}`}>
          {/* Pass state and toggle function to SideBar */}
          <SideBar
            systemInfo={systemInfo}
            isCollapsed={isSidebarCollapsed}
            toggleSidebar={toggleSidebar}
          />

          {/* Main chat area switches based on route */}
          <div className="chat-area-wrapper"> {/* Added wrapper for chat content */}
            <Routes>
              {/* Redirect root path to a new chat */}
              <Route path="/" element={<RedirectToNewChat />} />
              {/* Route for specific chat sessions */}
              <Route path="/chat/:chatId" element={<ChatPage systemInfo={systemInfo} />} />
              {/* Optional: Add a 404 or default route */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
