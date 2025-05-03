import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';

// Component to handle redirection from root
function RedirectToNewChat() {
  const navigate = useNavigate();
  useEffect(() => {
    // Redirect to a new chat session when accessing the root path
    navigate(`/chat/${uuidv4()}`, { replace: true });
  }, [navigate]);
  return null; // Render nothing while redirecting
}

export default RedirectToNewChat;
