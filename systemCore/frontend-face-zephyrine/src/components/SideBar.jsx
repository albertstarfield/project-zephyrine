// externalAnalyzer/frontend-face-zephyrine/src/components/SideBar.jsx
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import '../styles/components/_sidebar.css';
import InfoModal from './InfoModal';
import PropTypes from 'prop-types';

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
  chats,
  onRenameChat,
  onDeleteChat,
  onActivateVoiceMode,
}) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [infoModalMessage, setInfoModalMessage] = useState(null);
  const { signOut } = useAuth();
  const [isUserMenuExpanded, setIsUserMenuExpanded] = useState(true);
  const [editingChatId, setEditingChatId] = useState(null);
  const [editText, setEditText] = useState('');
  const inputRef = useRef(null);
  const [isHovering, setIsHovering] = useState(false);

  useEffect(() => {
    if (editingChatId && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editingChatId]);

  const handleNewChatClick = () => { if (onNewChat) onNewChat(); };
  const toggleUserMenu = () => { setIsUserMenuExpanded(!isUserMenuExpanded); };
  const handleLogout = () => {
    window.location.href = 'https://www.youtube.com/watch?v=xvFZjo5PgG0';
  };

  const handleInfoClick = () => {
    setInfoModalMessage("This setting is accessible from the config.py backend (Not the Frontend Backend) or the systemCore/engineMain itself! It is not a standard on openAI API. Thank you");
  };

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
      {infoModalMessage && (
        <InfoModal
          message={infoModalMessage}
          onClose={() => setInfoModalMessage(null)}
        />
      )}
      
      <button className="sidebar-hamburger-button" onClick={toggleSidebar} aria-label="Toggle sidebar">
        <HamburgerIcon />
      </button>

      {/* MODIFIED: Updated class logic for sidebar states */}
      <aside
        className={`sidebar ${isCollapsed ? (isHovering ? "" : "sidebar--collapsed") : "sidebar--expanded-visual"}`}
        onMouseEnter={() => setIsHovering(true)}
        onMouseLeave={() => setIsHovering(false)}
      >
        <button className="sidebar-toggle-button sidebar-toggle-desktop" onClick={toggleSidebar} title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}>
          {isCollapsed ? <ExpandIcon /> : <CollapseIcon />}
        </button>

        <div className="sidebar-top-actions">
          <button className="sidebar-button new-chat-button" onClick={handleNewChatClick} title={isCollapsed ? 'New Chat' : ''}>
            <svg viewBox="0 0 24 24" className="sidebar-icon" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor">
              <g><path d="M12 7V17M7 12H17" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"></path><path opacity="0.5" d="M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z" strokeWidth="2"></path></g>
            </svg>
            {/* Show text only if not collapsed OR if collapsed but hovering (expanded visual) */}
            {!(isCollapsed && !isHovering) && <span>New Chat</span>} 
          </button>
          {!(isCollapsed && !isHovering) && ( /* Show only if not collapsed visually */
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
          {!(isCollapsed && !isHovering) && <h4>History</h4>} {/* Show only if not collapsed visually */}
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
                    {/* Show full title if not collapsed visually, otherwise show placeholder/clipped */}
                    <span className="history-item-title">{!(isCollapsed && !isHovering) ? chat.title : (chat.title?.substring(0, 2).toUpperCase() || '...')}</span>
                    {!(isCollapsed && !isHovering) && ( /* Show actions only if not collapsed visually */
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
          
          <Link to="/" className="sidebar-button MainPageChat" title={isCollapsed ? 'MainPageChat' : ''}>
          <svg viewBox="0 0 24 24" className="sidebar-icon" fill="none" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <path d="M7.45648 3.08984C4.21754 4.74468 2 8.1136 2 12.0004C2 13.6001 2.37562 15.1121 3.04346 16.4529C3.22094 16.8092 3.28001 17.2165 3.17712 17.6011L2.58151 19.8271C2.32295 20.7934 3.20701 21.6775 4.17335 21.4189L6.39939 20.8233C6.78393 20.7204 7.19121 20.7795 7.54753 20.957C8.88836 21.6248 10.4003 22.0005 12 22.0005C16.8853 22.0005 20.9524 18.4973 21.8263 13.866C20.1758 15.7851 17.7298 17.0004 15 17.0004C10.0294 17.0004 6 12.971 6 8.00045C6 6.18869 6.53534 4.50197 7.45648 3.08984Z" fill="currentColor"></path> <path opacity="0.5" d="M21.8263 13.8655C21.9403 13.2611 22 12.6375 22 12C22 6.47715 17.5228 2 12 2C10.4467 2 8.97611 2.35415 7.66459 2.98611C7.59476 3.01975 7.52539 3.05419 7.45648 3.08939C6.53534 4.50152 6 6.18824 6 8C6 12.9706 10.0294 17 15 17C17.7298 17 20.1758 15.7847 21.8263 13.8655Z" fill="currentColor"></path> </g></svg>
            {!(isCollapsed && !isHovering) && <span>Main Page</span>}
          </Link>

          <button className="sidebar-button voice-assistant-button" title={isCollapsed ? 'Voice Assistant Mode' : ''} onClick={onActivateVoiceMode}>
            <svg viewBox="0 0 24 24" fill="none" className="sidebar-icon" xmlns="http://www.w3.org/2000/svg">
              <g>
                <path d="M3 8.25V15.75" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path>
                <path opacity="0.4" d="M7.5 5.75V18.25" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path>
                <path d="M12 3.25V20.75" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path>
                <path opacity="0.4" d="M16.5 5.75V18.25" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path>
                <path d="M21 8.25V15.75" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path>
              </g>
            </svg>
            {!(isCollapsed && !isHovering) && <span>Voice Assistant</span>}
          </button>
            
          <Link to="/images" className="sidebar-button image-generation-button" title={isCollapsed ? 'Image Generation' : ''}>
              <svg viewBox="0 0 24 24" fill="none" className='sidebar-icon' xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <path d="M18.5116 10.0767C18.5116 10.8153 17.8869 11.4142 17.1163 11.4142C16.3457 11.4142 15.7209 10.8153 15.7209 10.0767C15.7209 9.33801 16.3457 8.7392 17.1163 8.7392C17.8869 8.7392 18.5116 9.33801 18.5116 10.0767Z" fill="currentColor"></path> <path fill-rule="evenodd" clip-rule="evenodd" d="M18.0363 5.53205C16.9766 5.39548 15.6225 5.39549 13.9129 5.39551H10.0871C8.37751 5.39549 7.02343 5.39548 5.9637 5.53205C4.87308 5.6726 3.99033 5.96873 3.29418 6.63601C2.59803 7.30329 2.28908 8.14942 2.14245 9.19481C1.99997 10.2106 1.99999 11.5085 2 13.1472V13.2478C1.99999 14.8864 1.99997 16.1843 2.14245 17.2001C2.28908 18.2455 2.59803 19.0916 3.29418 19.7589C3.99033 20.4262 4.87307 20.7223 5.9637 20.8629C7.02344 20.9994 8.37751 20.9994 10.0871 20.9994H13.9129C15.6225 20.9994 16.9766 20.9994 18.0363 20.8629C19.1269 20.7223 20.0097 20.4262 20.7058 19.7589C21.402 19.0916 21.7109 18.2455 21.8575 17.2001C22 16.1843 22 14.8864 22 13.2478V13.1472C22 11.5085 22 10.2106 21.8575 9.19481C21.7109 8.14942 21.402 7.30329 20.7058 6.63601C20.0097 5.96873 19.1269 5.6726 18.0363 5.53205ZM6.14963 6.8576C5.21373 6.97821 4.67452 7.2044 4.28084 7.58175C3.88716 7.95911 3.65119 8.47595 3.52536 9.37303C3.42443 10.0926 3.40184 10.9919 3.3968 12.1682L3.86764 11.7733C4.99175 10.8305 6.68596 10.8846 7.74215 11.897L11.7326 15.7219C12.1321 16.1049 12.7611 16.1571 13.2234 15.8457L13.5008 15.6589C14.8313 14.7626 16.6314 14.8664 17.8402 15.9092L20.2479 17.9862C20.3463 17.7222 20.4206 17.4071 20.4746 17.0219C20.6032 16.1056 20.6047 14.8977 20.6047 13.1975C20.6047 11.4972 20.6032 10.2893 20.4746 9.37303C20.3488 8.47595 20.1128 7.95911 19.7192 7.58175C19.3255 7.2044 18.7863 6.97821 17.8504 6.8576C16.8944 6.73441 15.6343 6.73298 13.8605 6.73298H10.1395C8.36575 6.73298 7.10559 6.73441 6.14963 6.8576Z" fill="currentColor"></path> <g opacity="0.5"> <path d="M17.0866 2.61039C16.2268 2.49997 15.1321 2.49998 13.7675 2.5H10.6778C9.31314 2.49998 8.21844 2.49997 7.35863 2.61039C6.46826 2.72473 5.72591 2.96835 5.13712 3.53075C4.79755 3.8551 4.56886 4.22833 4.41309 4.64928C4.91729 4.41928 5.48734 4.28374 6.12735 4.20084C7.21173 4.06037 8.5973 4.06038 10.3466 4.06039H14.2615C16.0108 4.06038 17.3963 4.06037 18.4807 4.20084C19.0397 4.27325 19.5453 4.38581 20.0003 4.56638C19.8457 4.17917 19.6253 3.83365 19.3081 3.53075C18.7193 2.96835 17.977 2.72473 17.0866 2.61039Z" fill="currentColor"></path> </g> </g></svg>
              {!(isCollapsed && !isHovering) && <span>Image Generation</span>}
          </Link>

          <Link to="/knowledge-tuning" className="sidebar-button finetuning-button" title={isCollapsed ? 'Finetuning Knowledge Learning' : ''}>
              <svg viewBox="0 0 24 24" fill="none" className="sidebar-icon" xmlns="http://www.w3.org/2000/svg">
                <g>
                  <path d="M21.4707 19V5C21.4707 3 20.4707 2 18.4707 2H14.4707C12.4707 2 11.4707 3 11.4707 5V19C11.4707 21 12.4707 22 14.4707 22H18.4707C20.4707 22 21.4707 21 21.4707 19Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"></path>
                  <path opacity="0.4" d="M11.4707 6H16.4707" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"></path>
                  <path opacity="0.4" d="M11.4707 18H15.4707" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"></path>
                  <path opacity="0.4" d="M11.4707 13.9502L16.4707 14.0002" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"></path>
                  <path opacity="0.4" d="M11.4707 10H14.4707" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"></path>
                  <path d="M5.4893 2C3.8593 2 2.5293 3.33 2.5293 4.95V17.91C2.5293 18.36 2.7193 19.04 2.9493 19.43L3.7693 20.79C4.7093 22.36 6.2593 22.36 7.1993 20.79L8.0193 19.43C8.2493 19.04 8.4393 18.36 8.4393 17.91V4.95C8.4393 3.33 7.1093 2 5.4893 2Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"></path>
                  <path opacity="0.4" d="M8.4393 7H2.5293" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"></path>
                </g>
              </svg>
              {!(isCollapsed && !isHovering) && <span>Knowledge Tuning</span>}
          </Link>

          <button className="sidebar-button user-button" title={isCollapsed ? user?.email : ''} onClick={handleInfoClick}>
            <svg viewBox="0 0 24 24" fill="none" className="sidebar-icon" xmlns="http://www.w3.org/2000/svg">
              <g>
                <path opacity="0.4" d="M12.1605 10.87C12.0605 10.86 11.9405 10.86 11.8305 10.87C9.45055 10.79 7.56055 8.84 7.56055 6.44C7.56055 3.99 9.54055 2 12.0005 2C14.4505 2 16.4405 3.99 16.4405 6.44C16.4305 8.84 14.5405 10.79 12.1605 10.87Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path>
                <path d="M7.1607 14.56C4.7407 16.18 4.7407 18.82 7.1607 20.43C9.9107 22.27 14.4207 22.27 17.1707 20.43C19.5907 18.81 19.5907 16.17 17.1707 14.56C14.4307 12.73 9.9207 12.73 7.1607 14.56Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path>
              </g>
            </svg>
            {!(isCollapsed && !isHovering) && <span className="user-email">{user?.email}</span>}
          </button>

          <button className="sidebar-button settings-button" title={isCollapsed ? 'Settings' : ''} onClick={handleInfoClick}>
            <svg viewBox="0 0 24 24" fill="none" className="sidebar-icon" xmlns="http://www.w3.org/2000/svg">
              <g>
                <path opacity="0.34" d="M12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15Z" stroke="currentColor" strokeWidth="1.5" strokeMiterlimit="10" strokeLinecap="round" strokeLinejoin="round"></path>
                <path d="M2 12.8799V11.1199C2 10.0799 2.85 9.21994 3.9 9.21994C5.71 9.21994 6.45 7.93994 5.54 6.36994C5.02 5.46994 5.33 4.29994 6.24 3.77994L7.97 2.78994C8.76 2.31994 9.78 2.59994 10.25 3.38994L10.36 3.57994C11.26 5.14994 12.74 5.14994 13.65 3.57994L13.76 3.38994C14.23 2.59994 15.25 2.31994 16.04 2.78994L17.77 3.77994C18.68 4.29994 18.99 5.46994 18.47 6.36994C17.56 7.93994 18.3 9.21994 20.11 9.21994C21.15 9.21994 22.01 10.0699 22.01 11.1199V12.8799C22.01 13.9199 21.16 14.7799 20.11 14.7799C18.3 14.7799 17.56 16.0599 18.47 17.6299C18.99 18.5399 18.68 19.6999 17.77 20.2199L16.04 21.2099C15.25 21.6799 14.23 21.3999 13.76 20.6099L13.65 20.4199C12.75 18.8499 11.27 18.8499 10.36 20.4199L10.25 20.6099C9.78 21.3999 8.76 21.6799 7.97 21.2099L6.24 20.2199C5.33 19.6999 5.02 18.5299 5.54 17.6299C6.45 16.0599 5.71 14.7799 3.9 14.7799C2.85 14.7799 2 13.9199 2 12.8799Z" stroke="currentColor" strokeWidth="1.5" strokeMiterlimit="10" strokeLinecap="round" strokeLinejoin="round"></path>
              </g>
            </svg>
            {!(isCollapsed && !isHovering) && <span>Settings</span>}
          </button>

          <button className="sidebar-button logout-button" onClick={handleLogout} title={isCollapsed ? 'Logout' : ''}>
            <svg viewBox="0 0 24 24" fill="none" className="sidebar-icon" xmlns="http://www.w3.org/2000/svg">
              <g>
                <path d="M15 3H7C5.89543 3 5 3.89543 5 5V19C5 20.1046 5.89543 21 7 21H15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path>
                <path d="M19 12L15 8M19 12L15 16M19 12H9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path>
              </g>
            </svg>
            {!(isCollapsed && !isHovering) && <span>Logout</span>}
          </button>

          
        </div>
      </aside>
    </>
  );
};

SideBar.propTypes = {
  systemInfo: PropTypes.object,
  isCollapsed: PropTypes.bool.isRequired,
  toggleSidebar: PropTypes.func.isRequired,
  user: PropTypes.object,
  chats: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      title: PropTypes.string.isRequired,
      updated_at: PropTypes.string.isRequired,
    })
  ).isRequired,
  onRenameChat: PropTypes.func.isRequired,
  onDeleteChat: PropTypes.func.isRequired,
  onActivateVoiceMode: PropTypes.func.isRequired,
};

SideBar.defaultProps = {
  systemInfo: {},
  user: null,
};

export default SideBar;