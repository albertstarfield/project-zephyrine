// ExternalAnalyzer/frontend-face-zephyrine/src/hooks/useChatHistory.js
import { useState, useCallback, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

// The hook now accepts a function to get the current WebSocket instance
export function useChatHistory(getWebSocket) { 
  const { user } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  // This function is now just a signal to App.jsx that a refresh is needed.
  const fetchChatHistory = useCallback((userIdToFetch) => {
    const currentUserId = userIdToFetch || user?.id;
    if (!currentUserId) {
      console.warn("useChatHistory: fetchChatHistory called, but no user ID is available.");
      setChatHistory([]);
      return;
    }
    const ws = getWebSocket && typeof getWebSocket === 'function' ? getWebSocket() : null;
    if (ws && ws.readyState === WebSocket.OPEN) {
      setIsLoadingHistory(true);
      console.log(`useChatHistory: Requesting chat list via WebSocket for user: ${currentUserId}`);
      ws.send(JSON.stringify({ 
        type: "get_chat_history_list", 
        payload: { userId: currentUserId } 
      }));
    } else {
      console.warn("useChatHistory: WebSocket not available to fetch chat history.");
    }
  }, [user, getWebSocket]);

  const handleRenameChat = useCallback(async (chatId, newTitle) => {
    if (!user?.id || !chatId || typeof newTitle !== 'string') return;

    const originalChat = chatHistory.find(chat => chat.id === chatId);
    const originalTitle = originalChat ? originalChat.title : '';
    setChatHistory(prev => prev.map(c => c.id === chatId ? { ...c, title: newTitle } : c));

    const ws = getWebSocket && typeof getWebSocket === 'function' ? getWebSocket() : null;
    if (ws && ws.readyState === WebSocket.OPEN) {
      console.log(`useChatHistory: Sending 'rename_chat' for ${chatId}`);
      ws.send(JSON.stringify({
        type: "rename_chat",
        payload: { chatId, newTitle, userId: user.id }
      }));
    } else {
      console.error("useChatHistory: WebSocket not available for rename_chat. Reverting optimistic update.");
      setChatHistory(prev => prev.map(c => c.id === chatId ? { ...c, title: originalTitle } : c));
    }
  }, [user, chatHistory, getWebSocket, setChatHistory]);

  const handleDeleteChat = useCallback(async (chatId) => {
    if (!user?.id || !chatId) return;

    const originalHistory = [...chatHistory];
    setChatHistory(prev => prev.filter(c => c.id !== chatId));
    
    const ws = getWebSocket && typeof getWebSocket === 'function' ? getWebSocket() : null;
    if (ws && ws.readyState === WebSocket.OPEN) {
      console.log(`useChatHistory: Sending 'delete_chat' for ${chatId}`);
      ws.send(JSON.stringify({
        type: "delete_chat",
        payload: { chatId, userId: user.id }
      }));
       if (location.pathname.includes(`/chat/${chatId}`)) {
        navigate('/', { replace: true });
      }
    } else {
      console.error("useChatHistory: WebSocket not available for delete_chat. Reverting optimistic update.");
      setChatHistory(originalHistory);
    }
  }, [user, chatHistory, location, navigate, getWebSocket, setChatHistory]);

  return {
    chatHistory,
    setChatHistory,
    isLoadingHistory,
    setIsLoadingHistory,
    fetchChatHistory,
    handleRenameChat,
    handleDeleteChat,
  };
}