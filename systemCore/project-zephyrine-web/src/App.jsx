import { useState, useEffect, useRef, useCallback } from 'react'; // Added useCallback
import {
  Routes,
  Route,
  useParams,
  useNavigate,
  Navigate,
  useLocation,
} from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';
import { supabase } from './utils/supabaseClient';
import { useAuth } from './contexts/AuthContext'; // Import useAuth
import Auth from './components/Auth'; // Import Auth component
import './styles/App.css';
import SideBar from './components/SideBar';
import ChatFeed from "./components/ChatFeed";
import InputArea from './components/InputArea';
import SystemOverlay from './components/SystemOverlay';
import './styles/ChatInterface.css';
import './styles/utils/_overlay.css';

// Component to handle individual chat sessions
// Pass user object and refreshHistory callback
function ChatPage({ systemInfo, user, refreshHistory }) {
  const { chatId } = useParams();
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [showPlaceholder, setShowPlaceholder] = useState(true); // Keep for welcome screen logic
  const [error, setError] = useState(null); // Add error state
  const bottomRef = useRef(null);

  // Fetch messages when chatId changes
  useEffect(() => {
    const fetchMessages = async () => {
      if (!chatId) return; // Don't fetch if chatId isn't available yet

      setError(null); // Clear previous errors
      setIsGenerating(true); // Show loading state while fetching

      const { data, error: fetchError } = await supabase
        .from("messages")
        .select("*")
        .eq("chat_id", chatId) // Filter by the current chat ID
        .order("created_at", { ascending: true });

      if (fetchError) {
        console.error("Error fetching messages:", fetchError);
        setError("Failed to load chat history.");
        setMessages([]); // Clear messages on error
      } else {
        setMessages(data || []); // Ensure data is not null
        setShowPlaceholder(data === null || data.length === 0); // Show placeholder only if no messages
      }
      setIsGenerating(false); // Hide loading state
    };

    fetchMessages();
    setInputValue(""); // Clear input when changing chats

    // Optional: Set up real-time subscription for this specific chat
    // const subscription = supabase
    //   .channel(`public:messages:chat_id=eq.${chatId}`)
    //   .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'messages', filter: `chat_id=eq.${chatId}` }, payload => {
    //     setMessages(prevMessages => [...prevMessages, payload.new]);
    //     setShowPlaceholder(false); // Hide placeholder on new message
    //   })
    //   .subscribe();

    // // Cleanup subscription on unmount or chatId change
    // return () => {
    //   supabase.removeChannel(subscription);
    // };
  }, [chatId]); // Re-run effect when chatId changes

  // Scroll to bottom when messages change
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleSendMessage = async (text) => {
    if (!text.trim() || !chatId) return; // Ensure text and chatId are present

    setError(null); // Clear previous errors
    const userMessageContent = text;
    setInputValue('');
    setShowPlaceholder(false);

    // Insert user message into Supabase, associating with the user
    const { data: insertedMessage, error: insertError } = await supabase
      .from('messages')
      .insert([
        {
          sender: 'user',
          content: userMessageContent,
          chat_id: chatId,
          user_id: user?.id, // Associate message with logged-in user
        },
      ])
      .select()
      .single();

    if (insertError) {
      console.error("Error sending message:", insertError);
      setError("Failed to send message.");
      setInputValue(userMessageContent); // Restore input value on error
      return;
    }

    // Update state with the actual message from Supabase (if not using real-time subscription)
    if (insertedMessage) {
      setMessages((prev) => [...prev, insertedMessage]);
      // If this was the first message, refresh the history list in the sidebar
      if (messages.length === 0) { // Check if messages *before* adding the new one was empty
        refreshHistory();
      }
    }


    // --- Assistant Response Logic Placeholder ---
    // setIsGenerating(true);
    // try {
    //   const assistantResponse = await getAssistantResponse(userMessageContent); // Call your backend
    //   if (assistantResponse) {
    //     const { data: insertedAssistant, error: assistantError } = await supabase
    //       .from("messages")
    //       .insert([{ sender: "assistant", content: assistantResponse, chat_id: chatId }])
    //       .select()
    //       .single();
    //     if (assistantError) throw assistantError;
    //     if (insertedAssistant) {
    //       setMessages((prev) => [...prev, insertedAssistant]);
    //     }
    //   } else {
    //     throw new Error("No response from assistant");
    //   }
    // } catch (err) {
    //   console.error("Error getting/saving assistant response:", err);
    //   setError("Failed to get or save assistant response.");
    // } finally {
    //   setIsGenerating(false);
    // }
    // --- End Placeholder ---
  };

  // Keep handleStopGeneration if you plan to implement actual async generation later
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
        {/* Display error message if any */}
        {error && <div className="error-message chat-error">{error}</div>}{" "}
        {/* Added chat-error class */}
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

// Main App component handles layout, routing, and authentication check
function App() {
  const { session, loading, user } = useAuth(); // Get auth state
  const [systemInfo, setSystemInfo] = useState({
    username: '',
    assistantName: "Adelaide Zephyrine Charlotte",
    cpuUsage: 0,
    cpuFree: 0,
    cpuCount: 0,
    threadsUtilized: 0,
    freeMem: 0,
    totalMem: 0,
    os: '',
  });
  const [stars, setStars] = useState([]);
  const location = useLocation();
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [chatHistory, setChatHistory] = useState([]); // State for chat history
  const navigate = useNavigate(); // Use navigate hook here

  // Fetch chat history for the user
  const fetchChatHistory = useCallback(async () => {
    if (!user) return;

    try {
      // Get distinct chat IDs associated with the user
      const { data: distinctChats, error: distinctError } = await supabase
        .from('messages')
        .select('chat_id', { count: 'exact', head: false }) // Select only chat_id, don't need count here really
        .eq('user_id', user.id);

      if (distinctError) {
        console.error("Error fetching distinct chat IDs:", distinctError);
        setChatHistory([]); // Reset on error
        return;
      }

      if (!distinctChats || distinctChats.length === 0) {
        setChatHistory([]); // No history yet
        return;
      }

      // Get unique chat IDs
      const uniqueChatIds = [...new Set(distinctChats.map(c => c.chat_id))];

      // Fetch the first message for each unique chat ID to generate a title
      const historyPromises = uniqueChatIds.map(async (chatId) => {
        const { data: firstMessage, error: firstMsgError } = await supabase
          .from('messages')
          .select('content, created_at')
          .eq('chat_id', chatId)
          .order('created_at', { ascending: true })
          .limit(1)
          .maybeSingle(); // Use maybeSingle to handle potential null result gracefully

        if (firstMsgError) {
          console.error(`Error fetching first message for chat ${chatId}:`, firstMsgError);
          return null; // Skip this chat on error
        }

        // If a chat somehow has no messages (e.g., deleted), skip it
        if (!firstMessage) {
            return null;
        }

        return {
          id: chatId,
          // Generate title from first ~40 chars
          title: firstMessage.content.substring(0, 40) + (firstMessage.content.length > 40 ? '...' : ''),
          timestamp: firstMessage.created_at // Store timestamp for sorting
        };
      });

      const resolvedHistory = (await Promise.all(historyPromises))
                                .filter(item => item !== null); // Filter out any nulls from errors/empty chats

      // Sort by timestamp descending (most recent first)
      resolvedHistory.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

      setChatHistory(resolvedHistory);

    } catch (error) {
        console.error("Unexpected error fetching chat history:", error);
        setChatHistory([]); // Reset on unexpected error
    }
  }, [user]); // Dependency on user

  // Fetch history when user loads or changes
  useEffect(() => {
    fetchChatHistory();
  }, [fetchChatHistory]);

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

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
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

  // Function to handle creating a new chat
  const handleNewChat = () => {
    const newChatId = uuidv4();
    navigate(`/chat/${newChatId}`);
    // Optimistically add to history? Or wait for the first message?
    // Let's wait for the first message to be sent, fetchChatHistory will pick it up.
    // Alternatively, could add a placeholder immediately:
    // setChatHistory(prev => [{ id: newChatId, title: "New Chat", timestamp: new Date().toISOString() }, ...prev]);
    // For now, rely on fetchChatHistory triggered by message saving or page load.
  };

  // --- Rename Chat Handler ---
  const handleRenameChat = async (chatId, newTitle) => {
    // Optimistically update local state
    setChatHistory(prevHistory =>
      prevHistory.map(chat =>
        chat.id === chatId ? { ...chat, title: newTitle } : chat
      )
    );

    // TODO: Persist rename to backend.
    // This currently requires a schema change (e.g., a 'chats' table or a 'title' column).
    // Option 1: Add a 'chats' table (id, user_id, title, created_at) - Recommended
    // Option 2: Update the first message content (hacky)
    // Example (Option 2 - Hacky):
    /*
    try {
      const { data: firstMessage, error: fetchError } = await supabase
        .from('messages')
        .select('id')
        .eq('chat_id', chatId)
        .order('created_at', { ascending: true })
        .limit(1)
        .single();

      if (fetchError) throw fetchError;

      if (firstMessage) {
        const { error: updateError } = await supabase
          .from('messages')
          .update({ content: newTitle + "..." }) // Need a way to store original content if needed
          .eq('id', firstMessage.id);
        if (updateError) throw updateError;
      }
    } catch (error) {
      console.error("Error persisting rename:", error);
      // Optionally revert local state change on error
      fetchChatHistory(); // Re-fetch to revert
    }
    */
    console.log(`Rename chat ${chatId} to "${newTitle}" (Local state updated, backend persistence needed)`);
  };

  // --- Delete Chat Handler ---
  const handleDeleteChat = async (chatId) => {
    // Optional: Add confirmation dialog here
    // const confirmed = window.confirm("Are you sure you want to delete this chat?");
    // if (!confirmed) return;

    try {
      // Delete messages from Supabase
      const { error: deleteError } = await supabase
        .from('messages')
        .delete()
        .eq('chat_id', chatId);

      if (deleteError) {
        throw deleteError;
      }

      // Remove from local state
      setChatHistory(prevHistory => prevHistory.filter(chat => chat.id !== chatId));

      // If the user is currently viewing the deleted chat, navigate away
      if (location.pathname === `/chat/${chatId}`) {
        navigate('/', { replace: true }); // Navigate to root, which redirects to a new chat
      }

      console.log(`Deleted chat ${chatId}`);

    } catch (error) {
      console.error("Error deleting chat:", error);
      // Handle error display to user if needed
    }
  };


  // Show loading indicator during initial auth check
  if (loading) {
    return <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>Loading...</div>;
  }

  // If no session, show Auth component
  if (!session) {
    return <Auth />;
  }

  // If session exists, render the main app
  return (
    <div id="content">
      {/* Overlay for mobile sidebar - only shown when sidebar is open on mobile */}
      {!isSidebarCollapsed && (
        <div className="sidebar-overlay" onClick={toggleSidebar}></div>
      )}

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
        <div
          className={`main-content-area ${
            isSidebarCollapsed ? "main-content-area--sidebar-collapsed" : ""
          }`}
        >
          {/* Pass state, toggle function, user, and new chat handler to SideBar */}
          <SideBar
            systemInfo={systemInfo}
            isCollapsed={isSidebarCollapsed}
            toggleSidebar={toggleSidebar}
            user={user}
            onNewChat={handleNewChat}
            chatHistory={chatHistory}
            onRenameChat={handleRenameChat} // Pass rename handler
            onDeleteChat={handleDeleteChat} // Pass delete handler
          />

          {/* Main chat area switches based on route */}
          <div className="chat-area-wrapper">
            {" "}
            {/* Added wrapper for chat content */}
            <Routes>
              {/* Redirect root path to a new chat */}
              <Route path="/" element={<RedirectToNewChat />} />
              {/* Route for specific chat sessions */}
              <Route
                path="/chat/:chatId"
                // Pass user and fetchChatHistory callback to ChatPage
                // ChatPage can call fetchChatHistory after sending a message to update the sidebar
                element={<ChatPage systemInfo={systemInfo} user={user} refreshHistory={fetchChatHistory} />}
              />
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
