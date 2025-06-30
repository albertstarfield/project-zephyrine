// externalAnalyzer/frontend-face-zephyrine/src/components/ChatPage.jsx
import React, { useState, useEffect, useRef, useCallback } from "react";
import { useParams } from "react-router-dom";
import PropTypes from 'prop-types';
import { v4 as uuidv4 } from "uuid";
import ChatFeed from "./ChatFeed";
import InputArea from "./InputArea";
import "../styles/ChatInterface.css";
import "../styles/utils/_overlay.css";

const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL || "ws://localhost:3001";

function ChatPage({
  systemInfo = { assistantName: "Zephyrine" },
  user = null,
  refreshHistory = () => console.warn("ChatPage: 'refreshHistory' prop was called but not provided."),
  selectedModel = "default-model",
  updateSidebarHistory = () => console.warn("ChatPage: 'updateSidebarHistory' prop was called but not passed by App.jsx."),
  triggerSidebarRefresh = () => console.warn("ChatPage: 'triggerSidebarRefresh' prop was called but not passed by App.jsx.")
}) {
  const { chatId } = useParams();
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [showPlaceholder, setShowPlaceholder] = useState(false);
  const [error, setError] = useState(null);
  const [streamingAssistantMessage, setStreamingAssistantMessage] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [localCopySuccessId, setLocalCopySuccessId] = useState('');
  const bottomRef = useRef(null);
  const ws = useRef(null);
  const currentAssistantMessageId = useRef(null);
  const accumulatedContentRef = useRef("");
  const isConnected = useRef(false);
  const streamingStartTimeRef = useRef(0);
  const latestRequestRef = useRef(null); // <-- Add this line

  useEffect(() => {
    setIsLoadingHistory(true);
    setError(null);
    setMessages([]);
    setShowPlaceholder(false);

    const connectWebSocket = () => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
         console.log("ChatPage: WebSocket already connected. Requesting messages and sidebar history.");
         requestMessagesForChat(chatId);
         if (user && user.id) {
            ws.current.send(JSON.stringify({ type: 'get_chat_history_list', payload: { userId: user.id } }));
         }
         return;
      }

      console.log(`ChatPage: Attempting to connect WebSocket to ${WEBSOCKET_URL}...`);
      ws.current = new WebSocket(WEBSOCKET_URL);

      ws.current.onopen = () => {
        console.log("ChatPage: WebSocket Connected");
        isConnected.current = true;
        setError(null);
        requestMessagesForChat(chatId);
        if (user && user.id) {
          console.log("ChatPage: Requesting chat history list for sidebar (new WS connection).");
          ws.current.send(JSON.stringify({ type: 'get_chat_history_list', payload: { userId: user.id } }));
        }
      };

      ws.current.onmessage = (event) => {
        handleWebSocketMessage(event.data);
      };

      ws.current.onerror = (err) => {
        console.error("ChatPage: WebSocket Error:", err);
        setError("WebSocket connection error. Please try refreshing.");
        isConnected.current = false;
        setIsLoadingHistory(false);
        setShowPlaceholder(true);
      };

      ws.current.onclose = () => {
        console.log("ChatPage: WebSocket Disconnected");
        isConnected.current = false;
        if (isLoadingHistory) {
           setError("WebSocket disconnected while loading history.");
           setIsLoadingHistory(false);
           setShowPlaceholder(true);
        }
      };
    };

    const requestMessagesForChat = (currentChatId) => {
       if (ws.current && ws.current.readyState === WebSocket.OPEN && currentChatId) {
           console.log(`ChatPage: Requesting messages for chat ID: ${currentChatId}`);
           ws.current.send(JSON.stringify({
               type: "get_messages",
               payload: { chatId: currentChatId }
           }));
           setIsLoadingHistory(true);
       } else if (!currentChatId) {
            console.warn("ChatPage: Cannot request messages: No chat ID.");
            setIsLoadingHistory(false);
            setShowPlaceholder(true);
       } else {
           console.warn("ChatPage: Cannot request messages: WebSocket not connected.");
           setError("WebSocket not connected. Cannot load messages.");
           setIsLoadingHistory(false);
           setShowPlaceholder(true);
       }
    };

    connectWebSocket();

    return () => {
      console.log("ChatPage: Closing WebSocket connection for chatId:", chatId);
      ws.current?.close();
      isConnected.current = false;
    };
  }, [chatId, user]);

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, streamingAssistantMessage]);

  const handleWebSocketMessage = useCallback((data) => {
    try {
      const message = JSON.parse(data);

      if ((message.type === 'chunk' || message.type === 'end') && message.payload.optimisticMessageId !== latestRequestRef.current) {
        // This message is from an old, cancelled stream. Ignore it completely.
        return; 
      }

      switch (message.type) {
        case "chat_history":
            if (message.payload.chatId === chatId) {
                setMessages(message.payload.messages || []);
                setShowPlaceholder(!message.payload.messages || message.payload.messages.length === 0);
            }
            setIsLoadingHistory(false);
            break;
        case "chat_history_error":
            console.error("ChatPage: Error loading chat history from backend:", message.message || message.payload?.error);
            setError(`Failed to load chat history: ${message.message || message.payload?.error}`);
            setMessages([]);
            setShowPlaceholder(true);
            setIsLoadingHistory(false);
            break;
        
        // --- START: Added Case to Handle Backend Errors ---
        case "error":
            console.error("ChatPage: Received error from backend:", message.payload?.message);
            setError(message.payload?.message || "An unknown error occurred on the backend.");
            setIsGenerating(false);
            setStreamingAssistantMessage(null);
            break;
        // --- END: Added Case ---

        case "chunk":
          setIsGenerating(true);
          const contentChunk = message.payload.content;
          accumulatedContentRef.current += contentChunk;

          const currentTime = Date.now();
          const elapsedTime = currentTime - streamingStartTimeRef.current;
          const currentChars = accumulatedContentRef.current.length;
          let tokensPerSecond = 0;
          if (elapsedTime > 0) {
            tokensPerSecond = (currentChars / elapsedTime) * 1000;
          }

          

          const currentContent = accumulatedContentRef.current;
          const hasOpenThink = currentContent.includes("<think>") && !currentContent.includes("</think>");
          const isCurrentlyThinking = hasOpenThink;
          
          setStreamingAssistantMessage((prev) => {
            const newId = prev?.id || currentAssistantMessageId.current || `temp-stream-${uuidv4()}`;
            if (!currentAssistantMessageId.current && !prev?.id) {
                currentAssistantMessageId.current = newId;
            }
            return {
              id: newId,
              sender: "assistant",
              content: accumulatedContentRef.current,
              chat_id: chatId,
              created_at: prev?.created_at || new Date().toISOString(),
              isLoading: true,
              tokensPerSecond: tokensPerSecond.toFixed(1),
              isThinking: isCurrentlyThinking,
            };
          });
          break;
        case "end":
          setIsGenerating(false);
          const finalContent = accumulatedContentRef.current;
          const hasOpenThinkFinal = finalContent.includes("<think>") && !finalContent.includes("</think>");
          const isCurrentlyThinkingFinal = hasOpenThinkFinal;

          if (finalContent && currentAssistantMessageId.current) {
            const finalMessage = {
              id: currentAssistantMessageId.current,
              sender: "assistant",
              content: finalContent,
              chat_id: chatId,
              created_at: streamingAssistantMessage?.created_at || new Date().toISOString(),
              isLoading: false,
              isThinking: isCurrentlyThinkingFinal,
            };
            setMessages((prev) => {
                const existing = prev.find(msg => msg.id === finalMessage.id);
                if (existing) {
                    return prev.map(msg => msg.id === finalMessage.id ? finalMessage : msg);
                } else {
                    return [...prev, finalMessage];
                }
            });
            const userMessagesInHistory = messages.filter(m => m.sender === 'user' && !m.id?.startsWith('temp-')).length;
            if (userMessagesInHistory === 1 && messages.filter(m => m.sender === 'assistant' && !m.isLoading).length === 0) {
                 console.log("ChatPage: First assistant response complete, calling triggerSidebarRefresh.");
                 triggerSidebarRefresh();
            }
          }
          setStreamingAssistantMessage(null);
          accumulatedContentRef.current = "";
          currentAssistantMessageId.current = null;
          streamingStartTimeRef.current = 0;
          break;
        case "message_status_update":
            setMessages(prevMessages => 
                prevMessages.map(msg => 
                    msg.id === message.payload.id ? { ...msg, status: message.payload.status } : msg
                )
            );
            break;
        case "title_updated":
          console.log("ChatPage: Received title_updated:", message.payload);
          triggerSidebarRefresh(); 
          break;
        case "chat_history_list":
          console.log("ChatPage: Received chat_history_list for sidebar", message.payload.chats);
          updateSidebarHistory(message.payload.chats || []); 
          break;
        case "message_saved":
            break;
        case "message_updated":
            console.log(`ChatPage: Message updated confirmation received: ID ${message.payload.id}`);
             if (refreshHistory) refreshHistory();
            break;
        case "message_save_error":
            console.error("ChatPage: Backend error saving message:", message.message || message.payload?.error);
            setError(`Failed to save message: ${message.message || message.payload?.error}`);
            break;
        case "message_update_error":
            console.error("ChatPage: Backend error updating message:", message.message || message.payload?.error);
            setError(`Failed to update message: ${message.message || message.payload?.error}`);
            break;
        case "stopped":
            console.log("ChatPage: Backend confirmed generation stopped.");
            break;
        default:
          console.warn("ChatPage: Unknown WebSocket message type:", message.type, message);
      }
    } catch (error) {
      console.error("ChatPage: Failed to parse WebSocket message or handle:", error, "Raw data:", data);
      setError("Received invalid data from server.");
      setIsGenerating(false);
      setStreamingAssistantMessage(null);
      accumulatedContentRef.current = "";
      currentAssistantMessageId.current = null;
      setIsLoadingHistory(false);
    }
  }, [chatId, refreshHistory, messages, streamingAssistantMessage, updateSidebarHistory, triggerSidebarRefresh, user]);

  const handleEditSave = async (messageId, newContent) => {
    if (isGenerating) return;
     if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      setError("WebSocket is not connected. Cannot save edit.");
      return;
    }
    setError(null);
    const originalMessages = [...messages]; 
    const editedMessageIndex = messages.findIndex((msg) => msg.id === messageId);

    if (editedMessageIndex === -1 || messages[editedMessageIndex].sender !== "user") {
      setError("Only existing user messages can be edited.");
      return;
    }

    const historyBeforeEdit = messages.slice(0, editedMessageIndex)
        .filter(m => !m.isLoading && !m.id?.startsWith('temp-'))
        .map(m => ({ sender: m.sender, content: m.content }));
    
    const editedUserMessageForLlm = { sender: "user", content: newContent };
    const messagesForLlmRegen = [...historyBeforeEdit, editedUserMessageForLlm];

    setMessages(prevMessages => {
        const updatedMessages = prevMessages.map(msg => 
            msg.id === messageId ? { ...msg, content: newContent, created_at: new Date().toISOString() } : msg
        );
        return updatedMessages.slice(0, editedMessageIndex + 1); 
    });
    setShowPlaceholder(false);

    try {
        const editPayload = {
            messageIdToUpdateInDB: messageId,
            newContent: newContent,
            chatId: chatId,
            userId: user?.id,
            historyForRegen: messagesForLlmRegen.slice(-20),
            model: selectedModel || "default",
        };
        console.log("ChatPage: Sending WS 'edit_message' message:", editPayload);
        ws.current.send(JSON.stringify({ type: "edit_message", payload: editPayload }));
        setIsGenerating(true);

        accumulatedContentRef.current = "";
        currentAssistantMessageId.current = `temp-assistant-${uuidv4()}`;
        setStreamingAssistantMessage({
            id: currentAssistantMessageId.current,
            sender: "assistant",
            content: "",
            chat_id: chatId,
            created_at: new Date().toISOString(),
            isLoading: true,
        });

    } catch (sendError) {
        console.error("ChatPage: WebSocket send error during edit:", sendError);
        setError("Failed to send edit request to server.");
        setIsGenerating(false);
        setStreamingAssistantMessage(null);
        setMessages(originalMessages);
    }
  };

  // --- START: Reordered Functions ---

  const handleStopGeneration = useCallback((isSilent = false) => {
    if (!isGenerating) return;
    console.log("ChatPage: Stopping generation...");
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "stop", payload: { chatId } }));
    }
    setIsGenerating(false);

    if (!isSilent && streamingAssistantMessage && accumulatedContentRef.current && currentAssistantMessageId.current) {
      const finalPartialMessage = {
        ...streamingAssistantMessage,
        content: accumulatedContentRef.current,
        isLoading: false,
      };
      setMessages((prev) => {
          const existing = prev.find(msg => msg.id === finalPartialMessage.id);
          if (existing) {
              return prev.map(msg => msg.id === finalPartialMessage.id ? finalPartialMessage : msg);
          } else {
              return [...prev, finalPartialMessage];
          }
      });
    }
    
    setStreamingAssistantMessage(null);
    accumulatedContentRef.current = "";
    currentAssistantMessageId.current = null;
  }, [isGenerating, streamingAssistantMessage, chatId]);

  const sendMessageOrRegenerate = useCallback(async (contentToSend, isRegeneration = false) => {
    if ((!contentToSend?.trim() && !fileData) || !ws.current || ws.current.readyState !== WebSocket.OPEN) {
      if (!ws.current || ws.current.readyState !== WebSocket.OPEN) setError("WebSocket is not connected.");
      return;
    }
    
    // Stop any currently active generation.
    if (isGenerating) {
      handleStopGeneration(true);
    }
    setError(null);

    // 1. Prepare all the data and new state *before* calling setMessages.
    const optimisticUserMessage = {
      id: uuidv4(),
      sender: "user",
      content: contentToSend + (fileData ? `\n[File: ${fileData.name}]` : ''),
      isLoading: true,
      status: 'sending',
    };

    latestRequestRef.current = optimisticUserMessage.id;

    const baseMessages = isRegeneration
      ? messages.slice(0, messages.findLastIndex(m => m.sender === 'assistant'))
      : messages;

    const newUiState = [...baseMessages, optimisticUserMessage];

    const historyForBackend = newUiState
      .map(m => ({ role: m.sender, content: m.content }));

    const messagePayload = {
      messages: historyForBackend,
      model: selectedModel,
      chatId: chatId,
      userId: user?.id,
      optimisticMessageId: optimisticUserMessage.id,
    };

    // 2. Perform all state updates and side effects sequentially.
    setMessages(newUiState);
    setInputValue("");
    setFileData(null);
    setShowPlaceholder(false);
    setIsGenerating(true);
    accumulatedContentRef.current = "";
    currentAssistantMessageId.current = `temp-assistant-${uuidv4()}`;
    setStreamingAssistantMessage({ id: currentAssistantMessageId.current, sender: "assistant", content: "", isLoading: true });

    ws.current.send(JSON.stringify({ type: "chat", payload: messagePayload }));

  }, [chatId, fileData, isGenerating, selectedModel, user, handleStopGeneration, messages]);

  // --- END: Reordered Functions ---

  const handleSendMessage = () => {
    // The guard should ONLY prevent sending empty messages.
    // The sendMessageOrRegenerate function will handle the isGenerating logic.
    if (!inputValue.trim() && !fileData) return;
    sendMessageOrRegenerate(inputValue, false);
  };

  const handleRegenerate = () => {
    if (isGenerating || !messages.length) return;
    const relevantMessages = messages.filter(m => !m.isLoading && !m.id?.startsWith('temp-'));
    const lastUserMessage = [...relevantMessages].reverse().find((msg) => msg.sender === "user");

    if (lastUserMessage) {
      sendMessageOrRegenerate(lastUserMessage.content, true);
    } else {
      setError("Cannot regenerate: No previous user message found.");
    }
  };
  
  const handleCopyToClipboard = async (text, messageId) => {
    try {
      await navigator.clipboard.writeText(text);
      setLocalCopySuccessId(messageId);
      setTimeout(() => setLocalCopySuccessId(""), 1500);
    } catch (err) {
      console.error("Failed to copy text: ", err);
      setError("Failed to copy text.");
    }
  };
  
  const handleExampleClick = (text) => {
    setInputValue(text);
  };

  const lastValidAssistantMessageIndex = messages
      .filter(m => !m.isLoading && !m.id?.startsWith('temp-'))
      .findLastIndex((msg) => msg.sender === "assistant");

  return (
    <>

      <div id="feed" className={showPlaceholder && !isLoadingHistory ? "welcome-screen" : ""}>
        {isLoadingHistory && (
            <div style={{ textAlign: 'center', padding: '20px', color: 'var(--primary-text)' }}>
                Loading chat history...
            </div>
        )}
        {!isLoadingHistory && (
            <ChatFeed
              messages={messages}
              streamingMessage={streamingAssistantMessage}
              showPlaceholder={showPlaceholder}
              isGenerating={isGenerating}
              onExampleClick={handleExampleClick}
              bottomRef={bottomRef}
              assistantName={systemInfo?.assistantName || "Assistant"}
              onCopy={handleCopyToClipboard}
              onRegenerate={handleRegenerate}
              onEditSave={handleEditSave}
              copySuccessId={localCopySuccessId}
              lastAssistantMessageIndex={lastValidAssistantMessageIndex}
            />
        )}
        {error && <div className="error-message chat-error">{error}</div>}
        <InputArea
          inputValue={inputValue}
          onInputChange={setInputValue}
          onSend={handleSendMessage}
          onStopGeneration={handleStopGeneration}
          onFileSelect={setFileData}
          selectedFile={fileData}
          selectedModel={selectedModel}
          disabled={isLoadingHistory || !isConnected.current} 
        />
      </div>
    </>
  );
}

ChatPage.propTypes = {
  systemInfo: PropTypes.shape({
    assistantName: PropTypes.string,
  }),
  user: PropTypes.shape({
    id: PropTypes.string,
  }),
  refreshHistory: PropTypes.func,
  selectedModel: PropTypes.string,
  updateSidebarHistory: PropTypes.func,
  triggerSidebarRefresh: PropTypes.func,
};

//export default ChatPage;
export default React.memo(ChatPage); // Wrap the export