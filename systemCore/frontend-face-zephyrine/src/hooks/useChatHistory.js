import { useState, useCallback, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
// Removed: import { supabase } from '../utils/supabaseClient'; // No longer using Supabase client
import { useAuth } from '../contexts/AuthContext'; // Still need user info (now the dummy user)

// NOTE: This hook now relies on external mechanisms (e.g., WebSocket messages sent
// by parent components or a dedicated service) to interact with the backend for
// fetching, renaming, and deleting chats. It primarily manages the local state
// and optimistic updates.

export function useChatHistory() {
  const { user } = useAuth(); // Get user from AuthContext (dummy user)
  const navigate = useNavigate();
  const location = useLocation();
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true); // Add loading state

  // --- Fetch Chat History ---
  // This function now primarily serves as a trigger/placeholder.
  // The actual data fetching needs to happen via WebSocket, likely initiated
  // by a parent component or service that then updates the chatHistory state.
  const fetchChatHistory = useCallback(async () => {
    if (!user) {
      setChatHistory([]);
      setIsLoadingHistory(false);
      return;
    }
    setIsLoadingHistory(true); // Indicate loading starts
    console.log(`useChatHistory: Triggering fetch for user: ${user.id}`);
    // --- TODO: Implement Backend Fetch ---
    // This should involve sending a WebSocket message like:
    // ws.current.send(JSON.stringify({ type: "get_chat_list", payload: { userId: user.id } }));
    //
    // And listening for a response message like "chat_list_update" elsewhere,
    // which would then call setChatHistory() with the received data.
    // For now, we'll just simulate finishing loading with potentially empty data.
    // In a real implementation, you'd wait for the WS response.
    // Example: If using a context/prop to update:
    // const historyFromBackend = await someFunctionToGetHistoryViaWS();
    // setChatHistory(historyFromBackend);

    // Simulate fetch completion (remove this in real implementation)
    // setChatHistory([]); // Or potentially load from localStorage as fallback?
    setIsLoadingHistory(false);

  }, [user]); // Dependency on user

  // Fetch history initially (trigger the process)
  useEffect(() => {
    fetchChatHistory();
  }, [fetchChatHistory]); // fetchChatHistory depends on user

  // --- Rename Chat ---
  const handleRenameChat = useCallback(async (chatId, newTitle) => {
    if (!user || !chatId || typeof newTitle === 'undefined') return;

    const originalHistory = [...chatHistory];
    // Optimistically update local state
    setChatHistory(prevHistory =>
      prevHistory.map(chat =>
        chat.id === chatId ? { ...chat, title: newTitle } : chat
      )
    );

    console.log(`useChatHistory: Renaming chat ${chatId} to "${newTitle}" (optimistic)`);
    // --- TODO: Send WebSocket Message to Backend ---
    // This should involve sending a message like:
    // ws.current.send(JSON.stringify({
    //   type: "rename_chat",
    //   payload: { chatId: chatId, newTitle: newTitle, userId: user.id }
    // }));
    //
    // Consider adding error handling based on potential backend failure response.
    // If backend sends "rename_chat_error", revert the state:
    // setChatHistory(originalHistory);

  }, [user, chatHistory]); // Depend on user and history for optimistic update


  // --- Delete Chat ---
  const handleDeleteChat = useCallback(async (chatId) => {
    if (!user || !chatId) return;

    const chatToDelete = chatHistory.find(chat => chat.id === chatId);
    const confirmed = window.confirm(`Are you sure you want to delete "${chatToDelete?.title || 'this chat'}" and all its messages?`);
    if (!confirmed) return;

    const originalHistory = [...chatHistory];
    // Optimistically update local state
    setChatHistory(prevHistory => prevHistory.filter(chat => chat.id !== chatId));

    console.log(`useChatHistory: Deleting chat ${chatId} (optimistic)`);
    // --- TODO: Send WebSocket Message to Backend ---
    // This should involve sending a message like:
    // ws.current.send(JSON.stringify({
    //   type: "delete_chat",
    //   payload: { chatId: chatId, userId: user.id }
    // }));
    //
    // Consider adding error handling based on potential backend failure response.
    // If backend sends "delete_chat_error", revert the state:
    // setChatHistory(originalHistory);


    // Navigate away if the current chat was deleted
    if (location.pathname === `/chat/${chatId}`) {
      console.log(`Navigating away from deleted chat ${chatId}`);
      navigate('/', { replace: true });
    }
  }, [user, chatHistory, location.pathname, navigate]); // Dependencies for optimistic update and navigation


  // Return state and functions needed by components
  // Note: Consumers of this hook now need to handle triggering the actual backend communication.
  return {
      chatHistory,
      isLoadingHistory, // Expose loading state
      fetchChatHistory, // Function to trigger refresh (needs implementation)
      handleRenameChat, // Handles optimistic update (needs WS call)
      handleDeleteChat  // Handles optimistic update (needs WS call)
  };
}