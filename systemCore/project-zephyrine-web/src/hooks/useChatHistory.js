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
      // Get distinct chat IDs associated with the user
      const { data: distinctChats, error: distinctError } = await supabase
        .from('messages')
        .select('chat_id') // Just need chat_id
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
          .maybeSingle();

        if (firstMsgError) {
          console.error(`Error fetching first message for chat ${chatId}:`, firstMsgError);
          return null; // Skip this chat on error
        }

        if (!firstMessage) {
            return null; // Skip if chat has no messages
        }

        return {
          id: chatId,
          title: firstMessage.content.substring(0, 40) + (firstMessage.content.length > 40 ? '...' : ''),
          timestamp: firstMessage.created_at
        };
      });

      const resolvedHistory = (await Promise.all(historyPromises))
                                .filter(item => item !== null);

      resolvedHistory.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      setChatHistory(resolvedHistory);

    } catch (error) {
        console.error("Unexpected error fetching chat history:", error);
        setChatHistory([]);
    }
  }, [user]); // Dependency on user

  // Fetch history initially and when user changes
  useEffect(() => {
    fetchChatHistory();
  }, [fetchChatHistory]); // fetchChatHistory depends on user, so this covers user changes

  const handleRenameChat = async (chatId, newTitle) => {
    // Optimistically update local state
    setChatHistory(prevHistory =>
      prevHistory.map(chat =>
        chat.id === chatId ? { ...chat, title: newTitle } : chat
      )
    );
    // TODO: Backend persistence logic remains here or could be further abstracted
    console.log(`Rename chat ${chatId} to "${newTitle}" (Local state updated, backend persistence needed)`);
    // Consider adding error handling and reverting state if backend fails
  };

  const handleDeleteChat = async (chatId) => {
    // Basic confirmation - consider a modal for better UX
    const confirmed = window.confirm("Are you sure you want to delete this chat and all its messages?");
    if (!confirmed) return;

    try {
      // Ensure user is available before attempting deletion
      if (!user) {
        console.error("Cannot delete chat: User not logged in.");
        // Optionally show an error message to the user
        return;
      }

      const { error: deleteError } = await supabase
        .from('messages')
        .delete()
        .eq('chat_id', chatId)
        .eq('user_id', user.id); // Ensure user can only delete their messages

      if (deleteError) {
        throw deleteError;
      }

      // Remove from local state *after* successful deletion
      setChatHistory(prevHistory => prevHistory.filter(chat => chat.id !== chatId));

      // Navigate away if the current chat was deleted
      if (location.pathname === `/chat/${chatId}`) {
        navigate('/', { replace: true });
      }
      console.log(`Deleted chat ${chatId}`);
    } catch (error) {
      console.error("Error deleting chat:", error);
      // Add user feedback for error (e.g., using a toast notification library)
      alert(`Error deleting chat: ${error.message}`); // Simple alert for now
    }
  };

  // Return state and functions needed by the component
  return { chatHistory, fetchChatHistory, handleRenameChat, handleDeleteChat };
}
