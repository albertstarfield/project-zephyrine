import { useState, useCallback, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { supabase } from '../utils/supabaseClient';
import { useAuth } from '../contexts/AuthContext'; // Need user info

export function useChatHistory() {
  const { user } = useAuth(); // Get user from AuthContext
  const navigate = useNavigate();
  const location = useLocation();
  const [chatHistory, setChatHistory] = useState([]);

  const fetchChatHistory = useCallback(async () => {
    if (!user) {
      setChatHistory([]); // Clear history if no user
      return;
    }

    try {
      // Fetch chat entries directly from the 'chats' table for the current user
      console.log(`Fetching chat history for user: ${user.id}`); // Add log
      const { data: chatsData, error: chatsError } = await supabase
        .from('chats')
        .select('id, title, created_at') // Select necessary fields
        .eq('user_id', user.id)
        .order('created_at', { ascending: false }); // Order by creation date, newest first

      if (chatsError) {
        console.error("Error fetching chats from chats table:", chatsError);
        setChatHistory([]); // Reset on error
        return;
      }

      // Format the data for the state
      const formattedHistory = chatsData.map(chat => ({
        id: chat.id,
        // Use the stored title, provide a fallback if somehow null/empty
        title: chat.title || `Chat ${chat.id.substring(0, 8)}...`,
        timestamp: chat.created_at
      }));

      // No need to sort again if ordered in the query
      setChatHistory(formattedHistory);

    } catch (error) {
        console.error("Unexpected error fetching chat history from chats table:", error);
        setChatHistory([]);
    }
  }, [user]); // Dependency on user

  // Fetch history initially and when user changes
  useEffect(() => {
    fetchChatHistory();
  }, [fetchChatHistory]); // fetchChatHistory depends on user, so this covers user changes

  // Function to rename a chat title in the 'chats' table
  const handleRenameChat = async (chatId, newTitle) => {
    if (!user) return; // Need user

    // Optimistically update local state
    const originalHistory = [...chatHistory];
    setChatHistory(prevHistory =>
      prevHistory.map(chat =>
        chat.id === chatId ? { ...chat, title: newTitle } : chat
      )
    );

    try {
      const { error } = await supabase
        .from('chats')
        .update({ title: newTitle, updated_at: new Date().toISOString() }) // Update title and timestamp
        .eq('id', chatId)
        .eq('user_id', user.id); // Ensure user owns the chat

      if (error) throw error;
      console.log(`Renamed chat ${chatId} to "${newTitle}"`);
    } catch (error) {
      console.error("Error renaming chat:", error);
      // Revert optimistic update on error
      setChatHistory(originalHistory);
      alert(`Error renaming chat: ${error.message}`); // Simple feedback
    }
  };

  // Function to delete a chat from the 'chats' table (and messages via CASCADE)
  const handleDeleteChat = async (chatId) => {
    if (!user) return; // Need user

    const confirmed = window.confirm("Are you sure you want to delete this chat and all its messages?");
    if (!confirmed) return;

    // Optimistically update local state
    const originalHistory = [...chatHistory];
    setChatHistory(prevHistory => prevHistory.filter(chat => chat.id !== chatId));

    try {
      const { error } = await supabase
        .from('chats') // Delete the entry in the 'chats' table
        .delete()
        .eq('id', chatId)
        .eq('user_id', user.id); // Ensure user owns the chat

      if (error) throw error;

      // Messages should be deleted automatically due to the CASCADE constraint set up in SQL

      console.log(`Deleted chat ${chatId}`);

      // Navigate away if the current chat was deleted
      if (location.pathname === `/chat/${chatId}`) {
        navigate('/', { replace: true });
      }
    } catch (error) {
      console.error("Error deleting chat:", error);
      // Revert optimistic update on error
      setChatHistory(originalHistory);
      alert(`Error deleting chat: ${error.message}`); // Simple feedback
    }
  };

  // Return state and functions needed by the component
  return { chatHistory, fetchChatHistory, handleRenameChat, handleDeleteChat };
}
