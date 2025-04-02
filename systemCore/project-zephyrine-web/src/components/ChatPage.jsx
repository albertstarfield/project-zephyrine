import { useState, useEffect, useRef, useCallback } from "react";
import { useParams } from "react-router-dom";
import { v4 as uuidv4 } from "uuid";
import { supabase } from "../utils/supabaseClient";
import ChatFeed from "./ChatFeed";
import InputArea from "./InputArea";
import { Copy, RefreshCw } from "lucide-react"; // Added icons
import "../styles/ChatInterface.css";
import "../styles/utils/_overlay.css";

// Define WebSocket URL
const WEBSOCKET_URL = "ws://localhost:3001";

// Component to handle individual chat sessions
function ChatPage({ systemInfo, user, refreshHistory, selectedModel }) {
  const { chatId } = useParams();
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [showPlaceholder, setShowPlaceholder] = useState(true);
  const [error, setError] = useState(null);
  const [streamingAssistantMessage, setStreamingAssistantMessage] =
    useState(null);
  const [copySuccess, setCopySuccess] = useState(""); // State for copy feedback
  const bottomRef = useRef(null);
  const ws = useRef(null);
  const currentAssistantMessageId = useRef(null);
  const accumulatedContentRef = useRef("");

  // --- Function to Update Chat Title in DB ---
  const updateChatTitleInDb = async (receivedChatId, newTitle) => {
    if (!receivedChatId || !newTitle) return;
    console.log(`Updating title for chat ${receivedChatId} to: ${newTitle}`);
    try {
      const { error: updateError } = await supabase
        .from("chats") // Assuming your table is named 'chats'
        .update({ title: newTitle })
        .eq("id", receivedChatId);

      if (updateError) {
        throw updateError;
      }
      console.log(`Chat ${receivedChatId} title updated successfully.`);
      refreshHistory(); // Refresh the sidebar list
    } catch (error) {
      console.error("Error updating chat title:", error);
      // Optionally set an error state or notify the user
      setError("Failed to update chat title.");
    }
  };

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

    // --- WebSocket Connection Setup ---
    const connectWebSocket = () => {
      console.log("Attempting to connect WebSocket...");
      ws.current = new WebSocket(WEBSOCKET_URL);

      ws.current.onopen = () => {
        console.log("WebSocket Connected");
        setError(null); // Clear connection errors on successful connect
      };

      ws.current.onmessage = (event) => {
        handleWebSocketMessage(event.data);
      };

      ws.current.onerror = (err) => {
        console.error("WebSocket Error:", err);
        setError("WebSocket connection error. Please try refreshing.");
        // Consider adding reconnect logic here if needed
      };

      ws.current.onclose = () => {
        console.log("WebSocket Disconnected");
        // Consider adding reconnect logic or user notification
      };
    };

    connectWebSocket();
    // --- End WebSocket Setup ---

    // Cleanup WebSocket on unmount or chatId change
    return () => {
      ws.current?.close();
      // supabase.removeChannel(subscription); // If using Supabase subscription
    };
  }, [chatId]); // Re-run effect when chatId changes

  // Scroll to bottom when messages change
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, streamingAssistantMessage]); // Also scroll when streaming message updates

  // --- WebSocket Message Handler ---
  const handleWebSocketMessage = useCallback(
    (data) => {
      try {
        const message = JSON.parse(data);
        // console.log("WS Message Received:", message); // Debug log

        switch (message.type) {
          case "chunk":
            setIsGenerating(true);
            const contentChunk = message.payload.content;
            accumulatedContentRef.current += contentChunk; // Append to ref

            // Update UI state for streaming display
            setStreamingAssistantMessage((prev) => {
              // Ensure we have a streaming message object to update
              if (!prev && currentAssistantMessageId.current) {
                // Initialize if it's the first chunk for this message ID
                return {
                  id: currentAssistantMessageId.current,
                  sender: "assistant",
                  content: accumulatedContentRef.current, // Use accumulated content
                  chat_id: chatId,
                  created_at: new Date().toISOString(), // Consider using a fixed start time
                  isLoading: true,
                };
              } else if (prev) {
                // Update existing streaming message content
                return { ...prev, content: accumulatedContentRef.current };
              }
              return prev; // Should not happen if currentAssistantMessageId is set
            });
            break;
          case "end":
            setIsGenerating(false);
            const finalContent = accumulatedContentRef.current; // Get final content from ref

            if (finalContent && currentAssistantMessageId.current) {
              // Create the final message object
              const finalMessage = {
                ...streamingAssistantMessage, // Get base details like chatId, created_at
                id: currentAssistantMessageId.current, // Use the tracked ID
                content: finalContent,
                isLoading: false,
              };

              // Save to Supabase
              saveAssistantMessage(finalContent, finalMessage.id); // Pass ID for potential update

              // Add the final message to the main messages list
              setMessages((prev) => [...prev, finalMessage]);
            } else {
              console.log(
                "End event received but no content accumulated or ID tracked."
              );
            }

            // Reset for next message
            setStreamingAssistantMessage(null);
            accumulatedContentRef.current = "";
            currentAssistantMessageId.current = null;
            break;
          case "title": // Handle incoming title from backend
            console.log("Received title message:", message.payload);
            if (message.payload.chatId === chatId) {
              // Ensure title is for the current chat
              updateChatTitleInDb(
                message.payload.chatId,
                message.payload.title
              );
            } else {
              console.warn(
                `Received title for different chat (${message.payload.chatId}), ignoring.`
              );
            }
            break;
          case "title_updated": // Handle confirmation that title was saved
            console.log(
              "Received title_updated confirmation:",
              message.payload
            );
            if (message.payload.chatId === chatId) {
              console.log(
                "Refreshing history after title update confirmation."
              );
              refreshHistory(); // Refresh the sidebar list
            }
            break;
          case "error":
            console.error("WebSocket Server Error:", message.payload.error);
            setError(`Assistant error: ${message.payload.error}`);
            setIsGenerating(false);
            setStreamingAssistantMessage(null); // Clear any streaming state on error
            accumulatedContentRef.current = "";
            currentAssistantMessageId.current = null;
            break;
          default:
            console.warn("Unknown WebSocket message type:", message.type);
        }
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
        setError("Received invalid data from server.");
        setIsGenerating(false);
        setStreamingAssistantMessage(null);
        accumulatedContentRef.current = "";
        currentAssistantMessageId.current = null;
      }
    },
    [chatId, refreshHistory]
  );

  // --- Function to save assistant message to Supabase ---
  const saveAssistantMessage = async (content, tempId) => {
    if (!content || !chatId) {
      console.warn(
        "Attempted to save assistant message without content or chatId."
      );
      return; // Don't proceed if no content or chat ID
    }

    const { data: dbAssistantMessage, error: assistantSaveError } =
      await supabase
        .from("messages")
        .insert([
          {
            sender: "assistant",
            content: content,
            chat_id: chatId,
            user_id: user?.id, // Associate with user if logged in
          },
        ])
        .select()
        .single();

    if (assistantSaveError) {
      console.error("Error saving assistant message:", assistantSaveError);
      setError("Failed to save assistant response.");
      // Update the message in the list to show an error state
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === tempId ? { ...msg, error: "Failed to save" } : msg
        )
      );
    } else if (dbAssistantMessage) {
      // Replace the temporary message with the final one from the database
      setMessages((prev) => {
        const existingMessages = prev.filter((msg) => msg.id !== tempId); // Remove potential temp message
        return [...existingMessages, dbAssistantMessage]; // Add the final message
      });
      console.log("Assistant message saved:", dbAssistantMessage.id);

      // NOTE: Title generation logic is now moved to the backend service.
      // The backend will handle calling the AI and updating the 'chats' table.
      // We still need to refresh the history here if it was the first AI response
      // so the sidebar potentially picks up the new title generated by the backend.
      const isFirstAssistantResponse =
        prev.length === 1 && prev[0].sender === "user";
      if (isFirstAssistantResponse) {
        console.log(
          "First assistant response saved, refreshing history for potential title update."
        );
        refreshHistory();
      }
    }
  };

  // --- Send Message / Regenerate Handler (WebSocket) ---
  const sendMessageOrRegenerate = async (
    contentToSend,
    isRegeneration = false
  ) => {
    if (!chatId) {
      setError("Cannot send message: No active chat selected.");
      console.error("sendMessageOrRegenerate called without chatId");
      return;
    }
    if (!contentToSend.trim() || isGenerating) return;
    setError(null);
    setShowPlaceholder(false); // Hide placeholder on send/regen

    let currentMessages = [...messages]; // Copy current messages

    // --- 1. Prepare and Save User Message (if not regenerating) ---
    if (!isRegeneration) {
      setInputValue(""); // Clear input only for new messages
      const userMessageData = {
        sender: "user",
        content: contentToSend,
        chat_id: chatId,
        user_id: user?.id,
      };
      // Add user message optimistically to UI
      const optimisticUserMessage = {
        ...userMessageData,
        created_at: new Date().toISOString(),
        id: uuidv4(),
      };
      currentMessages = [...currentMessages, optimisticUserMessage];
      setMessages(currentMessages);

      // Save user message to DB (no need to await)
      supabase
        .from("messages")
        .insert([userMessageData])
        .select()
        .single()
        .then(({ data: dbUserMessage, error: insertError }) => {
          if (insertError) {
            console.error("Error saving user message:", insertError);
            setError("Failed to save your message.");
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === optimisticUserMessage.id
                  ? { ...msg, error: "Failed to save" }
                  : msg
              )
            );
          } else {
            console.log("User message saved:", dbUserMessage?.id);
            // Refresh history if it was the first message in this chat session (client-side check)
            if (messages.length === 0) refreshHistory();
          }
        });
    } else {
      // If regenerating, remove the last assistant message visually
      const lastAssistantMsgIndex = currentMessages.findLastIndex(
        (msg) => msg.sender === "assistant"
      );
      if (lastAssistantMsgIndex > -1) {
        // Optionally delete from DB or mark as replaced later
        currentMessages.splice(lastAssistantMsgIndex, 1);
        setMessages(currentMessages); // Update UI immediately
      }
    }

    // --- 2. Send Message via WebSocket ---
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      try {
        const messageToSend = {
          type: "chat",
          payload: {
            // Send history *up to the point of the message being sent/regenerated*
            messages: currentMessages
              .slice(-10)
              .map((m) => ({ sender: m.sender, content: m.content })),
            model: selectedModel,
            chatId: chatId,
            userId: user?.id, // Include the user ID
            // Include the first user message content if this is the first message
            // Note: The backend uses the *presence* of this field along with message history length to trigger title gen
            firstUserMessageContent:
              currentMessages.filter((m) => m.sender === "user").length === 1
                ? contentToSend
                : undefined,
          },
        };
        ws.current.send(JSON.stringify(messageToSend));
        setIsGenerating(true);

        // --- 3. Prepare Streaming State ---
        accumulatedContentRef.current = "";
        currentAssistantMessageId.current = `temp-assistant-${Date.now()}`;
        setStreamingAssistantMessage({
          id: currentAssistantMessageId.current,
          sender: "assistant",
          content: "",
          chat_id: chatId,
          created_at: new Date().toISOString(),
          isLoading: true,
        });
      } catch (sendError) {
        console.error("WebSocket send error:", sendError);
        setError("Failed to communicate with the assistant.");
        setIsGenerating(false);
        setStreamingAssistantMessage(null);
      }
    } else {
      setError("WebSocket is not connected. Cannot send message.");
      console.error(
        "WebSocket is not open. ReadyState:",
        ws.current?.readyState
      );
    }
  };

  // --- Edit Message Handler ---
  const handleEditSave = async (messageId, newContent) => {
    if (isGenerating) return; // Don't allow edits while generating

    setError(null);
    const editedMessageIndex = messages.findIndex((msg) => msg.id === messageId);

    if (editedMessageIndex === -1) {
      console.error("Cannot save edit: Message not found", messageId);
      setError("Failed to save edit: Message not found.");
      return;
    }

    // Ensure the message being edited is a user message
    if (messages[editedMessageIndex].sender !== 'user') {
        console.error("Cannot edit non-user message", messageId);
        setError("Only user messages can be edited.");
        return;
    }

    // 1. Prepare the updated message and the history up to that point
    const updatedMessage = { ...messages[editedMessageIndex], content: newContent };
    const historyUpToEdit = messages.slice(0, editedMessageIndex); // History *before* the edited message
    const messagesForRegeneration = [...historyUpToEdit, updatedMessage]; // Include the *updated* user message

    // 2. Update UI State: Show only messages up to the edited one
    setMessages(messagesForRegeneration);
    setShowPlaceholder(false); // Ensure placeholder is hidden

    // 3. Update the edited message in the database (async, don't block UI)
    supabase
      .from("messages")
      .update({ content: newContent })
      .eq("id", messageId)
      .then(({ error: updateError }) => {
        if (updateError) {
          console.error("Error updating message in DB:", updateError);
          // Optionally revert UI or show persistent error
          setError("Failed to save edit to database.");
        } else {
          console.log("Message updated successfully in DB:", messageId);
          // Potentially delete subsequent messages from DB here if needed
          // For now, we rely on the UI state update and regeneration overwriting
        }
      });

    // 4. Trigger regeneration using the history up to the edited message
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      try {
        const messageToSend = {
          type: "chat", // Use the same type as regular chat
          payload: {
            // Send history *up to and including the edited message*
            messages: messagesForRegeneration
              .slice(-10) // Limit history if needed
              .map((m) => ({ sender: m.sender, content: m.content })),
            model: selectedModel,
            chatId: chatId,
            userId: user?.id,
            // No need for firstUserMessageContent here as it's not the first message
          },
        };
        ws.current.send(JSON.stringify(messageToSend));
        setIsGenerating(true);

        // Prepare streaming state
        accumulatedContentRef.current = "";
        currentAssistantMessageId.current = `temp-assistant-${Date.now()}`;
        setStreamingAssistantMessage({
          id: currentAssistantMessageId.current,
          sender: "assistant",
          content: "",
          chat_id: chatId,
          created_at: new Date().toISOString(),
          isLoading: true,
        });
      } catch (sendError) {
        console.error("WebSocket send error during edit regeneration:", sendError);
        setError("Failed to communicate with the assistant after edit.");
        setIsGenerating(false);
        setStreamingAssistantMessage(null);
        // Consider how to handle UI state if regeneration fails (e.g., show error, revert?)
      }
    } else {
      setError("WebSocket is not connected. Cannot regenerate after edit.");
      console.error(
        "WebSocket is not open during edit regeneration. ReadyState:",
        ws.current?.readyState
      );
       // Consider how to handle UI state if regeneration fails
    }
  };
  // --- End Edit Message Handler ---


  // Specific handler for the form submission (uses the combined function)
  const handleSendMessage = (text) => {
    sendMessageOrRegenerate(text, false);
  };

  // Specific handler for the regenerate button (uses the combined function)
  const handleRegenerate = () => {
    if (isGenerating) return;
    // Find the last *user* message before the last *assistant* message
    const lastAssistantIndex = messages.findLastIndex(
      (msg) => msg.sender === "assistant"
    );
    const relevantHistory =
      lastAssistantIndex > -1
        ? messages.slice(0, lastAssistantIndex)
        : messages;
    const lastUserMessage = relevantHistory
      .slice()
      .reverse()
      .find((msg) => msg.sender === "user");

    if (lastUserMessage) {
      sendMessageOrRegenerate(lastUserMessage.content, true);
    } else {
      setError(
        "Cannot regenerate: No previous user message found to use as context."
      );
    }
  };

  // Handler for copying text
  const handleCopy = async (text, messageId) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopySuccess(messageId); // Set the ID of the message that was copied
      setTimeout(() => setCopySuccess(""), 1500); // Clear feedback after 1.5s
    } catch (err) {
      console.error("Failed to copy text: ", err);
      setError("Failed to copy text.");
    }
  };

  const handleStopGeneration = () => {
    console.log("Stopping generation...");
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "stop" }));
      console.log("Sent stop request to backend.");
    }
    setIsGenerating(false);
    if (
      streamingAssistantMessage &&
      streamingAssistantMessage.content &&
      currentAssistantMessageId.current
    ) {
      const finalPartialMessage = {
        ...streamingAssistantMessage,
        isLoading: false,
      };
      saveAssistantMessage(finalPartialMessage.content, finalPartialMessage.id);
      setMessages((prev) => {
        if (prev.some((msg) => msg.id === finalPartialMessage.id)) {
          return prev.map((msg) =>
            msg.id === finalPartialMessage.id ? finalPartialMessage : msg
          );
        }
        return [...prev, finalPartialMessage];
      });
    }
    setStreamingAssistantMessage(null);
    accumulatedContentRef.current = "";
    currentAssistantMessageId.current = null;
  };

  const handleExampleClick = (text) => {
    setInputValue(text);
  };

  // Find the index of the last assistant message for the regenerate button
  const lastAssistantMessageIndex = messages.findLastIndex(
    (msg) => msg.sender === "assistant"
  );

  return (
    <>
      {!showPlaceholder && (
        <div className="chat-model-selector">
          <span>{selectedModel}</span>
        </div>
      )}

      <div id="feed" className={showPlaceholder ? "welcome-screen" : ""}>
        {/* Pass handlers and state down to ChatFeed */}
        <ChatFeed
          messages={messages}
          streamingMessage={streamingAssistantMessage}
          showPlaceholder={showPlaceholder}
          isGenerating={isGenerating}
          onExampleClick={handleExampleClick}
          bottomRef={bottomRef}
          assistantName={systemInfo.assistantName}
          // Add props for copy/regenerate
          onCopy={handleCopy}
          onRegenerate={handleRegenerate}
          onEditSave={handleEditSave} // Pass the new handler
          copySuccessId={copySuccess}
          lastAssistantMessageIndex={lastAssistantMessageIndex}
        />
        {error && <div className="error-message chat-error">{error}</div>}
        <InputArea
          value={inputValue}
          onChange={setInputValue}
          onSend={handleSendMessage} // Use the specific handler for form submission
          onStop={handleStopGeneration}
          isGenerating={isGenerating}
        />
      </div>
    </>
  );
}

export default ChatPage;
