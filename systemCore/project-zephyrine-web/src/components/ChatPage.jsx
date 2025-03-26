import { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';
import { supabase } from '../utils/supabaseClient';
import ChatFeed from "./ChatFeed";
import InputArea from './InputArea';
import '../styles/ChatInterface.css'; // Keep relevant styles if needed
import '../styles/utils/_overlay.css'; // Keep relevant styles if needed

// Component to handle individual chat sessions
function ChatPage({ systemInfo, user, refreshHistory, selectedModel }) {
  const { chatId } = useParams();
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [showPlaceholder, setShowPlaceholder] = useState(true);
  const [error, setError] = useState(null);
  const [streamingAssistantMessage, setStreamingAssistantMessage] = useState(null); // State for the message being streamed
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
    // ... (subscription code commented out as in original)

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
  }, [messages, streamingAssistantMessage]); // Also scroll when streaming message updates

  const handleSendMessage = async (text) => {
    if (!text.trim() || !chatId || isGenerating) return; // Prevent sending while generating

    setError(null);
    const userMessageContent = text;
    setInputValue('');
    setShowPlaceholder(false); // Hide placeholder on send

    // --- 1. Add User Message ---
    const userMessage = {
      sender: 'user',
      content: userMessageContent,
      chat_id: chatId,
      user_id: user?.id,
    };
    // Optimistically add user message to UI
    const optimisticUserMessage = { ...userMessage, created_at: new Date().toISOString(), id: uuidv4() }; // Add temp ID and timestamp
    setMessages((prev) => [...prev, optimisticUserMessage]);

    // Save user message to DB (async, don't necessarily wait)
    supabase
      .from('messages')
      .insert([userMessage]) // Insert the original object without temp ID
      .select()
      .single()
      .then(({ data: dbUserMessage, error: insertError }) => {
        if (insertError) {
          console.error("Error saving user message:", insertError);
          setError("Failed to save your message.");
          // Optionally update the optimistic message to show an error state
          setMessages(prev => prev.map(msg => msg.id === optimisticUserMessage.id ? { ...msg, error: 'Failed to save' } : msg));
        } else {
          // Update the message in state with the ID from DB if needed (optional)
          // setMessages(prev => prev.map(msg => msg.id === optimisticUserMessage.id ? dbUserMessage : msg));
          console.log("User message saved:", dbUserMessage);
          // If this was the first message, refresh the history list in the sidebar
          if (messages.length === 0) { // Check if messages *before* adding the new one was empty
             refreshHistory();
          }
        }
      });


    // --- 2. Prepare for Assistant Response ---
    setIsGenerating(true);
    const assistantMessageId = uuidv4(); // Generate ID for the assistant message
    let fullAssistantResponse = "";

    // --- 2. Prepare Streaming State ---
    setStreamingAssistantMessage({
      id: assistantMessageId, // Use generated ID
      sender: 'assistant',
      content: '',
      chat_id: chatId,
      created_at: new Date().toISOString(),
      isLoading: true,
    });


    // --- 3. Call Backend API for Streaming ---
    try {
       const response = await fetch(`http://localhost:3001/api/chat`, {
         method: 'POST',
         headers: {
           'Content-Type': 'application/json',
         },
         body: JSON.stringify({
           messages: [...messages, { sender: 'user', content: userMessageContent }].map(m => ({ sender: m.sender, content: m.content })),
           model: selectedModel,
         }),
       });

       if (!response.ok) {
         const errorData = await response.json().catch(() => ({ error: 'Failed to fetch stream' }));
         throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
       }

       if (!response.body) {
         throw new Error('Response body is null');
       }

       const reader = response.body.getReader();
       const decoder = new TextDecoder();
       let buffer = '';

       while (true) {
         const { done, value } = await reader.read();
         if (done) {
           console.log('Stream finished.');
           break;
         }

         buffer += decoder.decode(value, { stream: true });
         const lines = buffer.split('\n');
         buffer = lines.pop() || '';

         for (const line of lines) {
           if (line.startsWith('data:')) {
             try {
               const jsonData = JSON.parse(line.substring(5).trim());
               if (jsonData.content) {
                 fullAssistantResponse += jsonData.content;
                 setStreamingAssistantMessage(prev => prev ? { ...prev, content: fullAssistantResponse } : null);
               } else if (jsonData.error) {
                 console.error("Error from stream:", jsonData.error);
                 setError(`Stream error: ${jsonData.error}`);
                 reader.cancel();
                 break;
               }
             } catch (e) {
               console.error('Error parsing SSE data:', e, 'Line:', line);
             }
           } else if (line.startsWith('event: error')) {
             // Handle explicit error event
           } else if (line.startsWith('event: end')) {
             console.log('Received end event from stream.');
             reader.cancel();
             break;
           }
         }
         if (reader.closed) {
            break;
         }
       }

       setIsGenerating(false);
       setStreamingAssistantMessage(prev => prev ? { ...prev, isLoading: false } : null);

       if (fullAssistantResponse.trim()) {
         console.log("Attempting to save full response:", fullAssistantResponse);
         const { data: dbAssistantMessage, error: assistantSaveError } = await supabase
           .from('messages')
           .insert([
             {
               sender: 'assistant',
               content: fullAssistantResponse,
               chat_id: chatId,
               user_id: user?.id, // Associate with user if logged in
             },
           ])
           .select()
           .single();

         if (assistantSaveError) {
           console.error("Error saving assistant message:", assistantSaveError);
           setError("Failed to save assistant response.");
           setStreamingAssistantMessage(prev => prev ? { ...prev, error: 'Failed to save' } : null);
         } else {
           setMessages(prev => [...prev, dbAssistantMessage]);
           setStreamingAssistantMessage(null);
           console.log("Assistant message saved:", dbAssistantMessage);
         }
       } else {
         console.log("No content received from stream, clearing placeholder.");
         setStreamingAssistantMessage(null);
       }

    } catch (err) {
      console.error("Error fetching or processing stream:", err);
      setError(`Failed to get response: ${err.message}`);
      setStreamingAssistantMessage(null);
      setIsGenerating(false);
    }
  };


  const handleStopGeneration = () => {
    // TODO: Implement AbortController logic to stop the fetch request
    console.log("Stopping generation - cancellation logic needed");
    setIsGenerating(false); // Reset UI state
    setStreamingAssistantMessage(null); // Clear any partial streaming message
  };

  const handleExampleClick = (text) => {
    setInputValue(text);
    // Optionally focus the input area here
  };

  return (
    <>
      {/* Model Selector Display */}
      {!showPlaceholder && (
        <div className="chat-model-selector">
          <span>{selectedModel}</span>
        </div>
      )}

      {/* Main feed and input area */}
      <div id="feed" className={showPlaceholder ? "welcome-screen" : ""}>
        <ChatFeed
          messages={messages}
          streamingMessage={streamingAssistantMessage}
          showPlaceholder={showPlaceholder}
          isGenerating={isGenerating}
          onExampleClick={handleExampleClick}
          bottomRef={bottomRef}
          assistantName={systemInfo.assistantName}
        />
        {error && <div className="error-message chat-error">{error}</div>}
        <InputArea
          value={inputValue}
          onChange={setInputValue}
          onSend={handleSendMessage}
          onStop={handleStopGeneration}
          isGenerating={isGenerating}
        />
      </div>
    </>
  );
}

export default ChatPage;
