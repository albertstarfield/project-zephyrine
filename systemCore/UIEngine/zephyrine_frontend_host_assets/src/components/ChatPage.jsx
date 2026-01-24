// externalAnalyzer/frontend-face-zephyrine/src/components/ChatPage.jsx
import React, { useState, useEffect, useRef, useCallback } from "react";
import { useParams } from "react-router-dom";
import PropTypes from 'prop-types';
import { v4 as uuidv4 } from "uuid";
import ChatFeed from "./ChatFeed";
import InputArea from "./InputArea";
import "../styles/ChatInterface.css";
import "../styles/utils/_overlay.css";

//const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL || "ws://localhost:3001";
import { FrontendBackendRecieve } from '../config'; // Import the helper
const backendHttpUrl = window.FrontendBackendRecieve || "http://localhost:3001";
//const WEBSOCKET_URL = backendHttpUrl.replace(/^http/, 'ws');
const WEBSOCKET_URL = backendHttpUrl.replace(/^http/, 'ws') + "/zepzepadaui";

/* ARCHITECTURAL NOTE: "Async Recieve / Not Idealistic Sent"
   ---------------------------------------------------------
   This component implements a "Fire-and-Forget" messaging model.
   
   1. SENT (The Trigger): 
      When the user clicks send, we capture the timestamp immediately (Client Time) 
      and render an optimistic bubble. We do NOT wait for the server to acknowledge 
      receipt before letting the user continue. The request is "thrown" to the backend.
   
   2. RECIEVED (The Eventual Consistency):
      The backend processes the message at its own pace (async). When it's done, 
      it emits a 'user_message_saved' event. The frontend listens for this 
      asynchronously.
      
   3. RECONCILIATION:
      When the 'received' event arrives, we locate the original "sent" bubble 
      (using the optimistic ID or content matching) and stamp it as 'delivered'.
      We do not block. We do not enforce a strict "Turn-Based" (flip-flop) lock.
      Multiple messages can be "in-flight" simultaneously.
*/

function ChatPage({
  systemInfo = { assistantName: "Zephyrine" },
  user = null,
  refreshHistory = () => console.warn("ChatPage: 'refreshHistory' prop was called but not provided."),
  selectedModel = "default-model",
  updateSidebarHistory = () => console.warn("ChatPage: 'updateSidebarHistory' prop was called but not passed by App.jsx."),
  triggerSidebarRefresh = () => console.warn("ChatPage: 'triggerSidebarRefresh' prop was called but not passed by App.jsx.")
}) {
  const { chatId } = useParams();
  const [allMessages, setAllMessages] = useState([]);
  const [visibleMessages, setVisibleMessages] = useState([]);
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 0 });
  const scrollContainerRef = useRef(null);
  const [inputValue, setInputValue] = useState("");
  const [isLoadingPrevious, setIsLoadingPrevious] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [showPlaceholder, setShowPlaceholder] = useState(false);
  const [error, setError] = useState(null);
  //const [streamingAssistantMessage, setStreamingAssistantMessage] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [localCopySuccessId, setLocalCopySuccessId] = useState('');
  const bottomRef = useRef(null);
  const ws = useRef(null);
  const currentAssistantMessageId = useRef(null);
  const accumulatedContentRef = useRef("");
  const isConnected = useRef(false);
  
  useEffect(() => {
    setIsLoadingHistory(true);
    setError(null);
    setAllMessages([]);
    setVisibleMessages([]);
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

  const handleScroll = () => {
    if (scrollContainerRef.current) {
      const { scrollTop } = scrollContainerRef.current;
      if (scrollTop === 0 && visibleRange.start > 0) {
        setIsLoadingPrevious(true);
        const newStart = Math.max(0, visibleRange.start - 5);
        const newVisibleMessages = allMessages.slice(newStart, visibleRange.end);
        const oldScrollHeight = scrollContainerRef.current.scrollHeight;

        setVisibleMessages(newVisibleMessages);
        setVisibleRange({ start: newStart, end: visibleRange.end });

        requestAnimationFrame(() => {
          const newScrollHeight = scrollContainerRef.current.scrollHeight;
          scrollContainerRef.current.scrollTop = newScrollHeight - oldScrollHeight;
          setIsLoadingPrevious(false);
        });
      }
    }
  };

  useEffect(() => {
    const scrollContainer = scrollContainerRef.current;
    if (scrollContainer) {
      scrollContainer.addEventListener('scroll', handleScroll);
      return () => {
        scrollContainer.removeEventListener('scroll', handleScroll);
      };
    }
  }, [handleScroll]);

  useEffect(() => {
    if (bottomRef.current && (visibleMessages[visibleMessages.length - 1]?.sender === 'user' || isGenerating)) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [visibleMessages, isGenerating]);

  const updateVisibleMessages = (newAllMessages) => {
    const messageCount = newAllMessages.length;
    const initialLoadCount = 5;
    const start = Math.max(0, messageCount - initialLoadCount);
    const end = messageCount;
    setVisibleMessages(newAllMessages.slice(start, end));
    setVisibleRange({ start, end });
  };

  const handleWebSocketMessage = useCallback((data) => {
    try {
      const message = JSON.parse(data);

      switch (message.type) {
        case "chat_history":
            if (message.payload.chatId === chatId) {
                const messages = message.payload.messages || [];
                setAllMessages(messages);
                const messageCount = messages.length;
                const initialLoadCount = 5;
                const start = Math.max(0, messageCount - initialLoadCount);
                const end = messageCount;
                setVisibleMessages(messages.slice(start, end));
                setVisibleRange({ start, end });
                setShowPlaceholder(messageCount === 0);
            }
            setIsLoadingHistory(false);
            break;
        case "chat_history_error":
            console.error("ChatPage: Error loading chat history from backend:", message.message || message.payload?.error);
            setError(`Failed to load chat history: ${message.message || message.payload?.error}`);
            setAllMessages([]);
            setVisibleMessages([]);
            setShowPlaceholder(true);
            setIsLoadingHistory(false);
            break;
        case "full_response":
        case "chat":
          setIsGenerating(false);
          const { 
            content: assistantContent, 
            optimisticMessageId, 
            id: assistantMessageId,
            userMessage 
          } = message.payload;

          if (assistantContent && assistantContent.includes("*NoResponseForThisQuery*")) {
              console.log("ChatPage: Received '*NoResponseForThisQuery*' flag. Silencing response.");
              if (optimisticMessageId) {
                  setAllMessages((prev) => {
                      const updatedMessages = [...prev];
                      let userIdx = updatedMessages.findIndex(m => m.id === optimisticMessageId);
                      
                      // Fallback: search by content + sending status
                      if (userIdx === -1) {
                         userIdx = updatedMessages.findIndex(m => 
                            m.sender === 'user' && m.status === 'sending' && userMessage && m.content === userMessage.content
                         );
                      }

                      if (userIdx !== -1) {
                          updatedMessages[userIdx] = {
                              ...updatedMessages[userIdx],
                              id: userMessage && userMessage.id ? userMessage.id : updatedMessages[userIdx].id,
                              status: 'delivered'
                          };
                          updateVisibleMessages(updatedMessages);
                          return updatedMessages;
                      }
                      return prev;
                  });
              }
              accumulatedContentRef.current = "";
              currentAssistantMessageId.current = null;
              return; 
          }

          setAllMessages((prev) => {
            let updatedMessages = [...prev];

            // Resolve the Pending User Message (sent via Optimistic UI)
            let userIdx = -1;
            if (optimisticMessageId) {
                userIdx = updatedMessages.findIndex(m => m.id === optimisticMessageId);
            }
            
            // Fallback strategy if ID is missing (find last sending message with same content)
            if (userIdx === -1 && userMessage) {
                userIdx = updatedMessages.findIndex(m => 
                    m.sender === 'user' && m.status === 'sending' && m.content === userMessage.content
                );
            }

            if (userIdx !== -1 && userMessage && userMessage.id) {
                updatedMessages[userIdx] = {
                  ...updatedMessages[userIdx],
                  id: userMessage.id,
                  status: 'delivered'
                  // We do NOT update created_at here to preserve the "clicked time"
                };
            } else if (userIdx !== -1) {
                updatedMessages[userIdx] = { ...updatedMessages[userIdx], status: 'delivered' };
            }

            const assistantExists = updatedMessages.some(m => m.id === assistantMessageId);
            
            if (!assistantExists) {
              updatedMessages.push({
                id: assistantMessageId || uuidv4(),
                sender: "assistant",
                content: assistantContent,
                chat_id: chatId,
                created_at: new Date().toISOString(),
                isLoading: false,
              });
            }

            updateVisibleMessages(updatedMessages);
            return updatedMessages;
          });

          accumulatedContentRef.current = "";
          currentAssistantMessageId.current = null;
          break;

        case "error":
            console.error("ChatPage: Received error from backend:", message.payload?.message);
            setError(message.payload?.message || "An unknown error occurred on the backend.");
            setIsGenerating(false);
            break;

        case "chunk":
          setIsGenerating(true);
          const contentChunk = message.payload.content;
          accumulatedContentRef.current += contentChunk;

          const currentContent = accumulatedContentRef.current;
          const displayContent = currentContent.replace(/<think>[\s\S]*?<\/think>/g, "");
          const hasOpenThink = currentContent.includes("<think>") && !currentContent.includes("</think>");
          
           if (typeof setStreamingAssistantMessage === 'function') {
              setStreamingAssistantMessage((prev) => {
                const newId = prev?.id || currentAssistantMessageId.current || `temp-stream-${uuidv4()}`;
                if (!currentAssistantMessageId.current && !prev?.id) {
                    currentAssistantMessageId.current = newId;
                }
                return {
                  id: newId,
                  sender: "assistant",
                  content: displayContent,
                  chat_id: chatId,
                  created_at: prev?.created_at || new Date().toISOString(),
                  isLoading: true,
                  isThinking: hasOpenThink,
                };
              });
           }
          break;
        case "end":
          setIsGenerating(false);
          const finalContent = accumulatedContentRef.current;

          if (finalContent.includes("*NoResponseForThisQuery*")) {
             console.log("ChatPage: Stream ended with NoResponse flag. Discarding.");
             accumulatedContentRef.current = "";
             currentAssistantMessageId.current = null;
             if (typeof setStreamingAssistantMessage === 'function') {
                setStreamingAssistantMessage(null);
             }
             return;
          }

          const hasOpenThinkFinal = finalContent.replace(/<think>[\s\S]*?<\/think>/g, "");
          const isCurrentlyThinkingFinal = hasOpenThinkFinal;

          if (finalContent && currentAssistantMessageId.current) {
            const finalMessage = {
              id: currentAssistantMessageId.current,
              sender: "assistant",
              content: finalContent,
              chat_id: chatId,
              created_at: new Date().toISOString(),
              isLoading: false,
              isThinking: isCurrentlyThinkingFinal, 
            };
            setAllMessages((prev) => {
                const existing = prev.find(msg => msg.id === finalMessage.id);
                let newMessages;
                if (existing) {
                    newMessages = prev.map(msg => msg.id === finalMessage.id ? finalMessage : msg);
                } else {
                    newMessages = [...prev, finalMessage];
                }
                updateVisibleMessages(newMessages);
                return newMessages;
            });
            // FIX 1: Safely handle integer IDs via String() cast
            const userMessagesInHistory = allMessages.filter(m => m.sender === 'user' && !String(m.id).startsWith('temp-')).length;
            if (userMessagesInHistory === 1 && allMessages.filter(m => m.sender === 'assistant' && !m.isLoading).length === 0) {
                 console.log("ChatPage: First assistant response complete, calling triggerSidebarRefresh.");
                 triggerSidebarRefresh();
            }
          }
          
          accumulatedContentRef.current = "";
          currentAssistantMessageId.current = null;
          break;
        case "user_message_saved":
          // ARCHITECTURAL NOTE: "Recieved (The Eventual Consistency)"
          setAllMessages(prev => {
              // 1. Primary Strategy: Find the local message by Optimistic ID
              let existingIndex = -1;
              if (message.payload.optimisticMessageId) {
                  existingIndex = prev.findIndex(m => m.id === message.payload.optimisticMessageId);
              }

              // 2. Fallback Strategy: Find by Content & Status
              if (existingIndex === -1) {
                  // Search backwards to find the most recent matching request
                  for (let i = prev.length - 1; i >= 0; i--) {
                      if (prev[i].sender === 'user' && 
                          prev[i].status === 'sending' && 
                          prev[i].content === message.payload.content) {
                          existingIndex = i;
                          break;
                      }
                  }
              }
              
              const savedUserMessage = {
                  id: message.payload.id,
                  sender: message.payload.sender,
                  content: message.payload.content,
                  chat_id: message.payload.chat_id,
                  created_at: message.payload.created_at, // Server time
                  isLoading: false,
                  status: 'delivered', 
              };

              let newAllMessages;
              if (existingIndex !== -1) {
                  // Found the pending message! Replace it.
                  newAllMessages = [...prev];
                  newAllMessages[existingIndex] = {
                      ...savedUserMessage,
                      // CRITICAL: Keep local timestamp to avoid visual jump
                      created_at: prev[existingIndex].created_at 
                  };
              } else {
                  // Not found (maybe a refresh happened?), append as new.
                  newAllMessages = [...prev, savedUserMessage];
              }
              
              updateVisibleMessages(newAllMessages);
              return newAllMessages;
          });
          break;
        case "title_updated":
          console.log("ChatPage: Received title_updated:", message.payload);
          triggerSidebarRefresh(); 
          break;
        case "chat_history_list":
          console.log("ChatPage: Received chat_history_list for sidebar", message.payload.chats);
          updateSidebarHistory(message.payload.chats || []); 
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
      
      accumulatedContentRef.current = "";
      currentAssistantMessageId.current = null;
      setIsLoadingHistory(false);
    }
  }, [chatId, refreshHistory, allMessages, updateSidebarHistory, triggerSidebarRefresh, user]);

  const handleEditSave = async (messageId, newContent) => {
    if (isGenerating) return;
     if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      setError("WebSocket is not connected. Cannot save edit.");
      return;
    }
    setError(null);
    const originalMessages = [...allMessages]; 
    const editedMessageIndex = allMessages.findIndex((msg) => msg.id === messageId);

    if (editedMessageIndex === -1 || allMessages[editedMessageIndex].sender !== "user") {
      setError("Only existing user messages can be edited.");
      return;
    }

    // FIX 2: Safely handle integer IDs here as well
    const historyBeforeEdit = allMessages.slice(0, editedMessageIndex)
        .filter(m => !m.isLoading && !String(m.id).startsWith('temp-'))
        .map(m => ({ sender: m.sender, content: m.content }));
    
    const editedUserMessageForLlm = { sender: "user", content: newContent };
    const messagesForLlmRegen = [...historyBeforeEdit, editedUserMessageForLlm];

    setAllMessages(prevMessages => {
        const updatedMessages = prevMessages.map(msg => 
            msg.id === messageId ? { ...msg, content: newContent, created_at: new Date().toISOString() } : msg
        );
        const newVisible = updatedMessages.slice(0, editedMessageIndex + 1);
        updateVisibleMessages(newVisible);
        return newVisible;
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
        if (typeof setStreamingAssistantMessage === 'function') {
            setStreamingAssistantMessage({
                id: currentAssistantMessageId.current,
                sender: "assistant",
                content: "",
                chat_id: chatId,
                created_at: new Date().toISOString(),
                isLoading: true,
            });
        }

    } catch (sendError) {
        console.error("ChatPage: WebSocket send error during edit:", sendError);
        setError("Failed to send edit request to server.");
        setIsGenerating(false);
        
        setAllMessages(originalMessages);
    }
  };

  const handleStopGeneration = useCallback((isSilent = false) => {
    if (!isGenerating) return;
    console.log("ChatPage: Stopping generation...");
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "stop", payload: { chatId } }));
    }
    setIsGenerating(false);
    
    accumulatedContentRef.current = "";
    currentAssistantMessageId.current = null;
  }, [isGenerating, chatId]);

  const sendMessageOrRegenerate = useCallback(async (contentToSend, isRegeneration = false) => {
    if ((!contentToSend?.trim() && !fileData) || !ws.current || ws.current.readyState !== WebSocket.OPEN) {
      if (!ws.current || ws.current.readyState !== WebSocket.OPEN) setError("WebSocket is not connected.");
      return;
    }
    
    setError(null);

    const tempOptimisticId = uuidv4(); 
    
    // --- KEY CHANGE: Capture the timestamp here ---
    const currentTimestamp = new Date().toISOString(); 

    // ARCHITECTURAL NOTE: "Sent (The Trigger)"
    // We display immediately (Optimistic UI) and throw the request to the server.
    const optimisticUserMessage = {
        id: tempOptimisticId,
        sender: "user",
        content: contentToSend + (fileData ? `\n[File: ${fileData.name}]` : ''),
        chat_id: chatId,
        created_at: currentTimestamp, // Use captured time
        status: 'sending',
    };

    if (!isRegeneration) {
        setAllMessages(prev => {
            const updated = [...prev, optimisticUserMessage];
            updateVisibleMessages(updated);
            return updated;
        });
    }

    const finalContent = contentToSend + (fileData ? `\n[File: ${fileData.name}]` : '');

    const baseMessages = isRegeneration
      ? allMessages.slice(0, allMessages.findLastIndex(m => m.sender === 'assistant'))
      : allMessages;

    const historyForBackend = [
        ...baseMessages.map(m => ({ role: m.sender, content: m.content })),
        { role: 'user', content: finalContent }
    ];

    const messagePayload = {
      messages: historyForBackend,
      model: selectedModel,
      chatId: chatId,
      userId: user?.id,
      optimisticMessageId: tempOptimisticId,
      created_at: currentTimestamp, 
      stream: false, 
    };

    setInputValue("");
    setFileData(null);
    setShowPlaceholder(false);
    setIsGenerating(true);
    accumulatedContentRef.current = "";
    
    ws.current.send(JSON.stringify({ type: "chat", payload: messagePayload }));

}, [chatId, fileData, isGenerating, selectedModel, user, handleStopGeneration, allMessages]);

  const handleSendMessage = () => {
    if (!inputValue.trim() && !fileData) return;
    sendMessageOrRegenerate(inputValue, false);
  };

  const handleRegenerate = () => {
    if (isGenerating || !allMessages.length) return;
    // FIX 3: Safely handle integer IDs here
    const relevantMessages = allMessages.filter(m => !m.isLoading && !String(m.id).startsWith('temp-'));
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

  // FIX 4: Safely handle integer IDs here
  const lastValidAssistantMessageIndex = allMessages
      .filter(m => !m.isLoading && !String(m.id).startsWith('temp-'))
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
              messages={visibleMessages}
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
              scrollContainerRef={scrollContainerRef}
              isLoadingPrevious={isLoadingPrevious}
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
          isGenerating={isGenerating}
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
export default React.memo(ChatPage);