// ExternalAnalyzer/frontend-face-zephyrine/src/hooks/useChatHistory.js
import { useState, useCallback, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

// getWebSocket: A function that returns the current WebSocket instance (e.g., () => ws.current from App.jsx)
export function useChatHistory(getWebSocket) { 
  const { user } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  const fetchChatHistory = useCallback(async (userIdToFetch) => {
    const currentUserId = userIdToFetch || user?.id;
    if (!currentUserId) {
      console.log("useChatHistory: Cannot fetch history, no user ID.");
      setChatHistory([]);
      setIsLoadingHistory(false);
      return;
    }

    const ws = getWebSocket && typeof getWebSocket === 'function' ? getWebSocket() : null;
    if (ws && ws.readyState === WebSocket.OPEN) {
      setIsLoadingHistory(true);
      console.log(`useChatHistory: Requesting chat history list for user: ${currentUserId}`);
      ws.send(JSON.stringify({ 
        type: "get_chat_history_list", 
        payload: { userId: currentUserId } 
      }));
      // Note: The actual setChatHistory(data) will be called by the component 
      // managing the WebSocket's onmessage handler (e.g., App.jsx or ChatPage.jsx)
      // after receiving the 'chat_history_list' message.
      // We can set a timeout to revert isLoadingHistory if no response is received,
      // or the parent component can call setIsLoadingHistory(false).
    } else {
      console.warn("useChatHistory: WebSocket not available or not open to fetch chat history.");
      // setChatHistory([]); // Optionally clear or leave as is
      setIsLoadingHistory(false); // Ensure loading stops if WS not available
    }
  }, [user, getWebSocket]);

  useEffect(() => {
    // Initial fetch is now more reliably triggered by ChatPage on WebSocket open.
    // This useEffect can be removed if App.jsx/ChatPage handles initial fetch.
    // Or, it can be a secondary trigger if `user` changes and `getWebSocket` is available.
    if (user?.id && getWebSocket && typeof getWebSocket() === 'object') { // Check if ws is available
      // fetchChatHistory(user.id); // Potentially redundant if ChatPage always fetches.
    } else if (!user?.id) {
      setChatHistory([]);
      setIsLoadingHistory(false);
    }
  }, [user, fetchChatHistory, getWebSocket]);


  const handleRenameChat = useCallback(async (chatId, newTitle) => {
    if (!user?.id || !chatId || typeof newTitle !== 'string') {
      console.warn("useChatHistory: Invalid parameters for rename chat.", { chatId, newTitle, userId: user?.id });
      return;
    }

    const originalChat = chatHistory.find(chat => chat.id === chatId);
    const originalTitle = originalChat ? originalChat.title : '';

    // Optimistic update
    setChatHistory(prevHistory =>
      prevHistory.map(chat =>
        chat.id === chatId ? { ...chat, title: newTitle, updated_at: new Date().toISOString() } : chat
      )
    );
    console.log(`useChatHistory: Renaming chat ${chatId} to "${newTitle}" (optimistic).`);

    const ws = getWebSocket && typeof getWebSocket === 'function' ? getWebSocket() : null;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: "rename_chat",
        payload: { chatId: chatId, newTitle: newTitle, userId: user.id }
      }));
      // The component handling ws.onmessage should listen for 'chat_renamed' or 'rename_chat_error'
      // and potentially call triggerSidebarRefresh or revert the optimistic update.
    } else {
      console.error("useChatHistory: WebSocket not available for rename_chat. Reverting optimistic update.");
      setChatHistory(prevHistory =>
        prevHistory.map(chat =>
          chat.id === chatId ? { ...chat, title: originalTitle } : chat // Revert
        )
      );
      // Optionally show an error to the user
    }
  }, [user, chatHistory, getWebSocket, setChatHistory]);


  const handleDeleteChat = useCallback(async (chatId) => {
    if (!user?.id || !chatId) {
      console.warn("useChatHistory: Invalid parameters for delete chat.", { chatId, userId: user?.id });
      return;
    }

    // Optional: Confirmation dialog (can be handled in SideBar.jsx before calling this)
    // const chatToDelete = chatHistory.find(chat => chat.id === chatId);
    // const confirmed = window.confirm(\`Are you sure you want to delete "${chatToDelete?.title || 'this chat'}"?\`);
    // if (!confirmed) return;

    const originalHistory = [...chatHistory];
    // Optimistic update
    setChatHistory(prevHistory => prevHistory.filter(chat => chat.id !== chatId));
    console.log(`useChatHistory: Deleting chat ${chatId} (optimistic).`);

    const ws = getWebSocket && typeof getWebSocket === 'function' ? getWebSocket() : null;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: "delete_chat",
        payload: { chatId: chatId, userId: user.id }
      }));
      // The component handling ws.onmessage should listen for 'chat_deleted' or 'delete_chat_error'
      // and potentially call triggerSidebarRefresh or revert optimistic update.
      // If current chat is deleted, navigation should occur.
      if (location.pathname.includes(`/chat/${chatId}`)) {
        console.log(`useChatHistory: Navigating away from deleted chat ${chatId}`);
        navigate('/', { replace: true }); // Navigate to a safe route
      }
    } else {
      console.error("useChatHistory: WebSocket not available for delete_chat. Reverting optimistic update.");
      setChatHistory(originalHistory); // Revert
      // Optionally show an error to the user
    }
  }, [user, chatHistory, location, navigate, getWebSocket, setChatHistory]);


  return {
    chatHistory,
    setChatHistory,     // For App.jsx/ChatPage.jsx to update from WS responses
    isLoadingHistory,
    setIsLoadingHistory, // Allow parent to control loading state
    fetchChatHistory,   // To trigger a get_chat_history_list message
    handleRenameChat,   // Now sends WS message
    handleDeleteChat,   // Now sends WS message
  };
}