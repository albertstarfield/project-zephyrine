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
    <div id="feed" className={showPlaceholder ? "welcome-screen" : ""}>
      <ChatFeed
        messages={messages}
        showPlaceholder={showPlaceholder}
        isGenerating={isGenerating}
        onExampleClick={handleExampleClick}
        bottomRef={bottomRef}
        assistantName={systemInfo.assistantName}
      />
      <InputArea
        value={inputValue}
        onChange={setInputValue}
        onSend={handleSendMessage}
        onStop={handleStopGeneration}
        isGenerating={isGenerating}
      />
    </div>
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
    navigate(`/chat/${uuidv4()}`);
  };


  return (
    <div id="content">
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
        {/* Pass handleNewChat and potentially chat history/list to SideBar */}
        {/* Note: handleNewChat needs access to navigate, might need adjustment */}
        <SideBar systemInfo={systemInfo} /* onNewChat={handleNewChat} - Re-enable later if needed */ />

        {/* Main content area switches based on route */}
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
  );
}

export default App;
