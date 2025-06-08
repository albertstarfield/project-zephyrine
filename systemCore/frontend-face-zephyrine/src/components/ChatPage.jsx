// ExternalAnalyzer/frontend-face-zephyrine/src/components/ChatPage.jsx
import React, { useState, useEffect, useRef, useCallback } from "react";
import { useParams } from "react-router-dom";
import PropTypes from 'prop-types';
import { v4 as uuidv4 } from "uuid";
import ChatFeed from "./ChatFeed";
import InputArea from "./InputArea";
import "../styles/ChatInterface.css";
import "../styles/utils/_overlay.css";

const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL || "ws://localhost:3001";

// Using JavaScript default parameters in the function signature.
// This is the modern way and replaces the static defaultProps block.
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
  const [localCopySuccessId, setLocalCopySuccessId] = useState('');
  const bottomRef = useRef(null);
  const ws = useRef(null);
  const currentAssistantMessageId = useRef(null);
  const accumulatedContentRef = useRef("");
  const isConnected = useRef(false);

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
        case "chunk":
          setIsGenerating(true);
          const contentChunk = message.payload.content;
          accumulatedContentRef.current += contentChunk;
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
            };
          });
          break;
        case "end":
          setIsGenerating(false);
          const finalContent = accumulatedContentRef.current;
          if (finalContent && currentAssistantMessageId.current) {
            const finalMessage = {
              id: currentAssistantMessageId.current,
              sender: "assistant",
              content: finalContent,
              chat_id: chatId,
              created_at: streamingAssistantMessage?.created_at || new Date().toISOString(),
              isLoading: false,
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
        case "error":
          console.error("ChatPage: WebSocket Server Error:", message.message || "Unknown server error");
          setError(`Assistant error: ${message.message || "Unknown server error"}`);
          setIsGenerating(false);
          setStreamingAssistantMessage(null);
          accumulatedContentRef.current = "";
          currentAssistantMessageId.current = null;
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
  }, [chatId, refreshHistory, messages, streamingAssistantMessage, updateSidebarHistory, triggerSidebarRefresh, user]); // CRITICAL FIX: Ensure all external props/state used are listed here

  const sendMessageOrRegenerate = async (contentToSend, isRegeneration = false) => {
    if (!chatId) {
      setError("Cannot send message: No active chat selected.");
      return;
    }
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      setError("WebSocket is not connected. Cannot send message.");
      return;
    }
    if (!contentToSend.trim() || (isGenerating && !isRegeneration)) return;
    setError(null);

    let currentMessagesForHistory = messages.filter(m => !m.isLoading && !m.id?.startsWith('temp-'));
    let userMessageId = null;
    let currentUserMessageForPayload = { sender: "user", content: contentToSend };

    if (!isRegeneration) {
      setInputValue("");
      const optimisticUserMessage = {
        sender: "user",
        content: contentToSend,
        chat_id: chatId,
        user_id: user?.id,
        created_at: new Date().toISOString(),
        id: `temp-user-${uuidv4()}`,
      };
      userMessageId = optimisticUserMessage.id;
      setMessages(prev => [...prev, optimisticUserMessage]); 
      setShowPlaceholder(false);
    } else {
      const lastUserMsgIndex = currentMessagesForHistory.findLastIndex(msg => msg.sender === "user");
       if (lastUserMsgIndex > -1) {
           currentUserMessageForPayload = {
              sender: "user",
              content: currentMessagesForHistory[lastUserMsgIndex].content 
           };
           currentMessagesForHistory = currentMessagesForHistory.slice(0, lastUserMsgIndex + 1);
       } else {
          setError("Cannot regenerate: No previous user message found in history.");
          return;
       }
    }
    
    const historyForBackendPayload = currentMessagesForHistory
        .slice(-20) 
        .map(m => ({ sender: m.sender || m.role, content: m.content }));

    let finalMessagesForPayload = [...historyForBackendPayload];
    if (!isRegeneration) {
        if (finalMessagesForPayload.length === 0 || 
            finalMessagesForPayload[finalMessagesForPayload.length - 1].content !== currentUserMessageForPayload.content ||
            finalMessagesForPayload[finalMessagesForPayload.length - 1].sender !== "user") {
            finalMessagesForPayload.push(currentUserMessageForPayload);
        }
    }
    
    try {
      const messagePayload = {
        messages: finalMessagesForPayload,
        model: selectedModel || "default",
        chatId: chatId,
        userId: user?.id,
        firstUserMessageContent:
          messages.filter((m) => m.sender === "user" && !m.id?.startsWith('temp-')).length === 0 && !isRegeneration
            ? contentToSend
            : undefined,
      };

      if (!messagePayload.messages || messagePayload.messages.length === 0) {
          console.error("ChatPage: Attempting to send empty messages payload!", messagePayload);
          setError("Error: Cannot send an empty message list to the assistant.");
          if (!isRegeneration && userMessageId) {
            setMessages(prev => prev.filter(m => m.id !== userMessageId));
          }
          return;
      }

      console.log("ChatPage: Sending WS 'chat' message:", JSON.stringify(messagePayload, null, 2));
      ws.current.send(JSON.stringify({ type: "chat", payload: messagePayload }));

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
      console.error("ChatPage: WebSocket send error:", sendError);
      setError("Failed to communicate with the assistant.");
      setIsGenerating(false);
      setStreamingAssistantMessage(null);
       if (!isRegeneration && userMessageId) {
         setMessages(prev => prev.filter(m => m.id !== userMessageId));
       }
    }
  };

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

  const handleSendMessage = (text) => {
    sendMessageOrRegenerate(text, false);
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

  const handleStopGeneration = () => {
    if (!isGenerating) return;
    console.log("ChatPage: Stopping generation...");
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "stop", payload: { chatId } }));
      console.log("ChatPage: Sent stop request to backend.");
    } else {
        console.warn("ChatPage: Cannot send stop request: WebSocket not connected.");
    }
    setIsGenerating(false);

    if (streamingAssistantMessage && accumulatedContentRef.current && currentAssistantMessageId.current) {
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
  };

  const handleExampleClick = (text) => {
    setInputValue(text);
  };

  const lastValidAssistantMessageIndex = messages
      .filter(m => !m.isLoading && !m.id?.startsWith('temp-'))
      .findLastIndex((msg) => msg.sender === "assistant");

  return (
    <>
      {(!showPlaceholder || messages.length > 0) && (
        <div className="chat-model-selector">
          <span>{selectedModel || 'Default Model'}</span>
        </div>
      )}

      <div id="feed" className={showPlaceholder && !isLoadingHistory ? "welcome-screen" : ""}>
        {isLoadingHistory && (
            <div style={{ textAlign: 'center', padding: '20px', color: 'var(--primary-text)' }}>
                Loading chat history...
            </div>
        )}
        {!isLoadingHistory && (
            <ChatFeed
              messages={messages}
              streamingMessage={streamingAssistantMessage ? { 
                  id: streamingAssistantMessage.id,
                  content: streamingAssistantMessage.content,
                  isLoading: streamingAssistantMessage.isLoading,
              } : null }
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
          isGenerating={isGenerating}
          selectedModel={selectedModel}
          disabled={isLoadingHistory || !isConnected.current}
        />
      </div>
    </>
  );
}

// Keep PropTypes for documentation and type checking
ChatPage.propTypes = {
  systemInfo: PropTypes.shape({ assistantName: PropTypes.string }),
  user: PropTypes.shape({ id: PropTypes.string }),
  refreshHistory: PropTypes.func,
  selectedModel: PropTypes.string,
  updateSidebarHistory: PropTypes.func,
  triggerSidebarRefresh: PropTypes.func,
};

export default ChatPage;