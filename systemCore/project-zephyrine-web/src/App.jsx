import { useState, useEffect, useRef } from "react";
import "./styles/App.css";
import SideBar from "./components/SideBar";
import ChatFeed from "./components/ChatFeed";
import InputArea from "./components/InputArea";
import SystemOverlay from "./components/SystemOverlay";

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

  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [showPlaceholder, setShowPlaceholder] = useState(true);
  const [stars, setStars] = useState([]);
  const bottomRef = useRef(null);

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

    // Simulate CPU and memory updates
    const interval = setInterval(() => {
      setSystemInfo((prev) => ({
        ...prev,
        cpuUsage: Math.random() * 0.5,
        cpuFree: 1 - Math.random() * 0.5,
        freeMem: Math.random() * 4 + 2, // Between 2-6GB free
        threadsUtilized: Math.floor(
          Math.random() * (navigator.hardwareConcurrency || 4)
        ),
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleSendMessage = (text) => {
    if (!text.trim()) return;

    // Hide placeholder when first message is sent
    if (showPlaceholder) {
      setShowPlaceholder(false);
    }

    // Add user message
    const userMessage = {
      id: Date.now(),
      sender: "user",
      content: text,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");

    // Simulate assistant response
    setIsGenerating(true);

    // Sample responses based on input
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
        return "I understand your message. How can I assist you further with this topic?";
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
        <SideBar systemInfo={systemInfo} />

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
      </div>
    </div>
  );
}

export default App;
