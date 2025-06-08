import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import '../styles/components/_sidebar.css';

// --- Icons ---
const CollapseIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
    <line x1="9" y1="3" x2="9" y2="21"></line>
  </svg>
);
const ExpandIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
    <line x1="15" y1="3" x2="15" y2="21"></line>
  </svg>
);
const HamburgerIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="3" y1="12" x2="21" y2="12"></line>
    <line x1="3" y1="6" x2="21" y2="6"></line>
    <line x1="3" y1="18" x2="21" y2="18"></line>
  </svg>
);
const EditIcon = () => (
  <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16" className="sidebar-action-icon edit-icon">
    <path d="M17.414 2.586a2 2 0 00-2.828 0L7 10.172V13h2.828l7.586-7.586a2 2 0 000-2.828z"></path>
    <path fillRule="evenodd" d="M2 6a2 2 0 012-2h4a1 1 0 010 2H4v10h10v-4a1 1 0 112 0v4a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" clipRule="evenodd"></path>
  </svg>
);
const DeleteIcon = () => (
  <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16" className="sidebar-action-icon delete-icon">
    <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd"></path>
  </svg>
);
const CheckIcon = () => (
  <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16" className="sidebar-action-icon confirm-icon">
      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"></path>
  </svg>
);
const CancelIcon = () => (
    <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16" className="sidebar-action-icon cancel-icon">
        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd"></path>
    </svg>
);

const SideBar = ({
  systemInfo,
  isCollapsed,
  toggleSidebar,
  user,
  onNewChat,
  chats, // <<<<< THIS IS THE ONLY CHANGE NEEDED. Changed from `chatHistory` to `chats`.
  onRenameChat,
  onDeleteChat,
  onActivateVoiceMode,
}) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { signOut } = useAuth();
  const [isUserMenuExpanded, setIsUserMenuExpanded] = useState(true);
  const [editingChatId, setEditingChatId] = useState(null);
  const [editText, setEditText] = useState('');
  const inputRef = useRef(null);

  useEffect(() => {
    if (editingChatId && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editingChatId]);

  const handleNewChatClick = () => { if (onNewChat) onNewChat(); };
  const toggleUserMenu = () => { setIsUserMenuExpanded(!isUserMenuExpanded); };
  const handleLogout = async () => { await signOut(); };

  const startEditing = (chat) => {
    setEditingChatId(chat.id);
    setEditText(chat.title);
  };
  const cancelEditing = () => {
    setEditingChatId(null);
    setEditText('');
  };
  const handleRename = (chatId) => {
    if (editText.trim() && onRenameChat) {
      onRenameChat(chatId, editText.trim());
    }
    cancelEditing();
  };
  const handleDelete = (chatId) => {
    if (onDeleteChat) {
      onDeleteChat(chatId);
    }
    if(location.pathname.includes(chatId)){
      navigate('/');
    }
  };
  const handleInputKeyDown = (event, chatId) => {
    if (event.key === 'Enter') handleRename(chatId);
    else if (event.key === 'Escape') cancelEditing();
  };
  const handleInputBlur = (chatId) => {
    setTimeout(() => { if (editingChatId === chatId) handleRename(chatId); }, 100);
  };

  const currentChatId = location.pathname.split('/chat/')[1];

  return (
    <>
      <button className="sidebar-hamburger-button" onClick={toggleSidebar} aria-label="Toggle sidebar">
        <HamburgerIcon />
      </button>

      <aside className={`sidebar ${isCollapsed ? "sidebar--collapsed" : ""}`}>
        <button className="sidebar-toggle-button sidebar-toggle-desktop" onClick={toggleSidebar} title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}>
          {isCollapsed ? <ExpandIcon /> : <CollapseIcon />}
        </button>

        <div className="sidebar-top-actions">
          <button className="sidebar-button new-chat-button" onClick={handleNewChatClick} title={isCollapsed ? 'New Chat' : ''}>
            <svg viewBox="0 0 24 24" className="sidebar-icon" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor">
              <g><path d="M12 7V17M7 12H17" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"></path><path opacity="0.5" d="M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z" strokeWidth="2"></path></g>
            </svg>
            {!isCollapsed && <span>New Chat</span>}
          </button>
          {!isCollapsed && (
            <div className="model-selector-container">
              <svg className="dropdown-arrow" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                 <path d="M7 10L12 15L17 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path>
              </svg>
              <select className="model-selector-dropdown" disabled>
                <option>Zephyrine Unified Model</option>
              </select>
            </div>
          )}
        </div>

        <nav className="sidebar-history">
          {!isCollapsed && <h4>History</h4>}
          <ul>
          {(chats || []).map(chat => (
              <li key={chat.id} title={isCollapsed ? chat.title : ""} className={`history-item ${editingChatId === chat.id ? 'editing' : ''} ${chat.id === currentChatId ? 'active' : ''}`}>
                {editingChatId === chat.id ? (
                  <div className="history-item-edit-container">
                    <input ref={inputRef} type="text" value={editText} onChange={(e) => setEditText(e.target.value)} onKeyDown={(e) => handleInputKeyDown(e, chat.id)} onBlur={() => handleInputBlur(chat.id)} className="history-item-input"/>
                    <div className="history-item-edit-actions">
                       <button onClick={() => handleRename(chat.id)} className="icon-button" title="Confirm rename"><CheckIcon /></button>
                       <button onClick={cancelEditing} className="icon-button" title="Cancel rename"><CancelIcon /></button>
                    </div>
                  </div>
                ) : (
                  <Link to={`/chat/${chat.id}`} className="history-item-button">
                    <span className="history-item-title">{!isCollapsed ? chat.title : (chat.title?.substring(0, 2).toUpperCase() || '...')}</span>
                    {!isCollapsed && (
                      <div className="history-item-actions">
                        <button onClick={(e) => { e.preventDefault(); e.stopPropagation(); startEditing(chat); }} className="icon-button" title="Rename chat"><EditIcon /></button>
                        <button onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleDelete(chat.id); }} className="icon-button" title="Delete chat"><DeleteIcon /></button>
                      </div>
                    )}
                  </Link>
                )}
              </li>
            ))}
          </ul>
        </nav>

        <div className="sidebar-spacer"></div>

        <div className="sidebar-bottom-actions">
          <button className="sidebar-button user-button" title={isCollapsed ? user?.email : ''} onClick={toggleUserMenu}>
            <svg viewBox="0 0 24 24" fill="none" className="sidebar-icon" xmlns="http://www.w3.org/2000/svg"><g><path opacity="0.4" d="M12.1605 10.87C12.0605 10.86 11.9405 10.86 11.8305 10.87C9.45055 10.79 7.56055 8.84 7.56055 6.44C7.56055 3.99 9.54055 2 12.0005 2C14.4505 2 16.4405 3.99 16.4405 6.44C16.4305 8.84 14.5405 10.79 12.1605 10.87Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path> <path d="M7.1607 14.56C4.7407 16.18 4.7407 18.82 7.1607 20.43C9.9107 22.27 14.4207 22.27 17.1707 20.43C19.5907 18.81 19.5907 16.17 17.1707 14.56C14.4307 12.73 9.9207 12.73 7.1607 14.56Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path> </g></svg>
            {!isCollapsed && <span className="user-email">{user?.email}</span>}
          </button>
          
          <button className="sidebar-button voice-assistant-button" title={isCollapsed ? 'Voice Assistant Mode' : ''} onClick={onActivateVoiceMode}>
            <svg viewBox="0 0 24 24" fill="none" className="sidebar-icon" xmlns="http://www.w3.org/2000/svg"><g><path d="M3 8.25V15.75" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path> <path opacity="0.4" d="M7.5 5.75V18.25" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path> <path d="M12 3.25V20.75" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path> <path opacity="0.4" d="M16.5 5.75V18.25" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path> <path d="M21 8.25V15.75" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path> </g></svg>
            {!isCollapsed && <span>Voice Assistant</span>}
          </button>

          <button className="sidebar-button image-generation-button" title={isCollapsed ? 'Image Generation' : ''} onClick={() => console.log("Image Generation clicked")}>
            <svg viewBox="0 0 24 24" fill="none" className="sidebar-icon" xmlns="http://www.w3.org/2000/svg" stroke="currentColor"><g><path d="M9 22H15C20 22 22 20 22 15V9C22 4 20 2 15 2H9C4 2 2 4 2 9V15C2 20 4 22 9 22Z" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path><path opacity="0.4" d="M9 10C10.1046 10 11 9.10457 11 8C11 6.89543 10.1046 6 9 6C7.89543 6 7 6.89543 7 8C7 9.10457 7.89543 10 9 10Z" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path><path opacity="0.4" d="M2.66992 18.9501L7.59992 15.6401C8.38992 15.1101 9.52992 15.1701 10.2399 15.7801L10.5699 16.0701C11.3499 16.7401 12.6099 16.7401 13.3899 16.0701L17.5499 12.5001C18.3299 11.8301 19.5899 11.8301 20.3699 12.5001L21.9999 13.9001" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path></g></svg>
            {!isCollapsed && <span>Image Generation</span>}
          </button>

          <button className="sidebar-button finetuning-button" title={isCollapsed ? 'Finetuning Knowledge Learning' : ''} onClick={() => console.log("Finetuning Knowledge Learning clicked")}>
            <svg viewBox="0 0 24 24" fill="none" className="sidebar-icon" xmlns="http://www.w3.org/2000/svg" stroke="currentColor"><g><path d="M21.4707 19V5C21.4707 3 20.4707 2 18.4707 2H14.4707C12.4707 2 11.4707 3 11.4707 5V19C11.4707 21 12.4707 22 14.4707 22H18.4707C20.4707 22 21.4707 21 21.4707 19Z" strokeWidth="1.5" strokeLinecap="round"></path><path opacity="0.4" d="M11.4707 6H16.4707" strokeWidth="1.5" strokeLinecap="round"></path><path opacity="0.4" d="M11.4707 18H15.4707" strokeWidth="1.5" strokeLinecap="round"></path><path opacity="0.4" d="M11.4707 13.9502L16.4707 14.0002" strokeWidth="1.5" strokeLinecap="round"></path><path opacity="0.4" d="M11.4707 10H14.4707" strokeWidth="1.5" strokeLinecap="round"></path><path d="M5.4893 2C3.8593 2 2.5293 3.33 2.5293 4.95V17.91C2.5293 18.36 2.7193 19.04 2.9493 19.43L3.7693 20.79C4.7093 22.36 6.2593 22.36 7.1993 20.79L8.0193 19.43C8.2493 19.04 8.4393 18.36 8.4393 17.91V4.95C8.4393 3.33 7.1093 2 5.4893 2Z" strokeWidth="1.5" strokeLinecap="round"></path><path opacity="0.4" d="M8.4393 7H2.5293" strokeWidth="1.5" strokeLinecap="round"></path></g></svg>
            {!isCollapsed && <span>Knowledge Tuning</span>}
          </button>
          
          <button className="sidebar-button logout-button" onClick={handleLogout} title={isCollapsed ? 'Logout' : ''}>
            <svg viewBox="0 0 24 24" className="sidebar-icon" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor"><g><path d="M15 3H7C5.89543 3 5 3.89543 5 5V19C5 20.1046 5.89543 21 7 21H15" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path><path d="M19 12L15 8M19 12L15 16M19 12H9" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path></g></svg>
            {!isCollapsed && <span>Logout</span>}
          </button>

          <button className="sidebar-button settings-button" title={isCollapsed ? 'Settings' : ''}>
            <svg viewBox="0 0 24 24" className="sidebar-icon" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor"><g><path opacity="0.34" d="M12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15Z" strokeWidth="1.5" strokeMiterlimit="10" strokeLinecap="round" strokeLinejoin="round"></path><path d="M2 12.8799V11.1199C2 10.0799 2.85 9.21994 3.9 9.21994C5.71 9.21994 6.45 7.93994 5.54 6.36994C5.02 5.46994 5.33 4.29994 6.24 3.77994L7.97 2.78994C8.76 2.31994 9.78 2.59994 10.25 3.38994L10.36 3.57994C11.26 5.14994 12.74 5.14994 13.65 3.57994L13.76 3.38994C14.23 2.59994 15.25 2.31994 16.04 2.78994L17.77 3.77994C18.68 4.29994 18.99 5.46994 18.47 6.36994C17.56 7.93994 18.3 9.21994 20.11 9.21994C21.15 9.21994 22.01 10.0699 22.01 11.1199V12.8799C22.01 13.9199 21.16 14.7799 20.11 14.7799C18.3 14.7799 17.56 16.0599 18.47 17.6299C18.99 18.5399 18.68 19.6999 17.77 20.2199L16.04 21.2099C15.25 21.6799 14.23 21.3999 13.76 20.6099L13.65 20.4199C12.75 18.8499 11.27 18.8499 10.36 20.4199L10.25 20.6099C9.78 21.3999 8.76 21.6799 7.97 21.2099L6.24 20.2199C5.33 19.6999 5.02 18.5299 5.54 17.6299C6.45 16.0599 5.71 14.7799 3.9 14.7799C2.85 14.7799 2 13.9199 2 12.8799Z" strokeWidth="1.5" strokeMiterlimit="10" strokeLinecap="round" strokeLinejoin="round"></path></g></svg>
            {!isCollapsed && <span>Settings</span>}
          </button>
        </div>
      </aside>
    </>
  );
};

export default SideBar;