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
          <svg viewBox="0 0 24 24" fill="none" className='sidebar-icon' xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <path opacity="0.4" d="M18.4698 16.83L18.8598 19.99C18.9598 20.82 18.0698 21.4 17.3598 20.97L13.1698 18.48C12.7098 18.48 12.2599 18.45 11.8199 18.39C12.5599 17.52 12.9998 16.42 12.9998 15.23C12.9998 12.39 10.5398 10.09 7.49985 10.09C6.33985 10.09 5.26985 10.42 4.37985 11C4.34985 10.75 4.33984 10.5 4.33984 10.24C4.33984 5.68999 8.28985 2 13.1698 2C18.0498 2 21.9998 5.68999 21.9998 10.24C21.9998 12.94 20.6098 15.33 18.4698 16.83Z" fill="currentColor"></path> <path d="M13 15.2298C13 16.4198 12.56 17.5198 11.82 18.3898C10.83 19.5898 9.26 20.3598 7.5 20.3598L4.89 21.9098C4.45 22.1798 3.89 21.8098 3.95 21.2998L4.2 19.3298C2.86 18.3998 2 16.9098 2 15.2298C2 13.4698 2.94 11.9198 4.38 10.9998C5.27 10.4198 6.34 10.0898 7.5 10.0898C10.54 10.0898 13 12.3898 13 15.2298Z" fill="currentColor"></path> </g></svg>
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
              <svg viewBox="0 0 24 24" fill="none" className='sidebar-icon' xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <path d="M18.5116 10.0767C18.5116 10.8153 17.8869 11.4142 17.1163 11.4142C16.3457 11.4142 15.7209 10.8153 15.7209 10.0767C15.7209 9.33801 16.3457 8.7392 17.1163 8.7392C17.8869 8.7392 18.5116 9.33801 18.5116 10.0767Z" fill="currentColor"></path> <path fill-rule="evenodd" clipRule="evenodd" d="M18.0363 5.53205C16.9766 5.39548 15.6225 5.39549 13.9129 5.39551H10.0871C8.37751 5.39549 7.02343 5.39548 5.9637 5.53205C4.87308 5.6726 3.99033 5.96873 3.29418 6.63601C2.59803 7.30329 2.28908 8.14942 2.14245 9.19481C1.99997 10.2106 1.99999 11.5085 2 13.1472V13.2478C1.99999 14.8864 1.99997 16.1843 2.14245 17.2001C2.28908 18.2455 2.59803 19.0916 3.29418 19.7589C3.99033 20.4262 4.87307 20.7223 5.9637 20.8629C7.02344 20.9994 8.37751 20.9994 10.0871 20.9994H13.9129C15.6225 20.9994 16.9766 20.9994 18.0363 20.8629C19.1269 20.7223 20.0097 20.4262 20.7058 19.7589C21.402 19.0916 21.7109 18.2455 21.8575 17.2001C22 16.1843 22 14.8864 22 13.2478V13.1472C22 11.5085 22 10.2106 21.8575 9.19481C21.7109 8.14942 21.402 7.30329 20.7058 6.63601C20.0097 5.96873 19.1269 5.6726 18.0363 5.53205ZM6.14963 6.8576C5.21373 6.97821 4.67452 7.2044 4.28084 7.58175C3.88716 7.95911 3.65119 8.47595 3.52536 9.37303C3.42443 10.0926 3.40184 10.9919 3.3968 12.1682L3.86764 11.7733C4.99175 10.8305 6.68596 10.8846 7.74215 11.897L11.7326 15.7219C12.1321 16.1049 12.7611 16.1571 13.2234 15.8457L13.5008 15.6589C14.8313 14.7626 16.6314 14.8664 17.8402 15.9092L20.2479 17.9862C20.3463 17.7222 20.4206 17.4071 20.4746 17.0219C20.6032 16.1056 20.6047 14.8977 20.6047 13.1975C20.6047 11.4972 20.6032 10.2893 20.4746 9.37303C20.3488 8.47595 20.1128 7.95911 19.7192 7.58175C19.3255 7.2044 18.7863 6.97821 17.8504 6.8576C16.8944 6.73441 15.6343 6.73298 13.8605 6.73298H10.1395C8.36575 6.73298 7.10559 6.73441 6.14963 6.8576Z" fill="currentColor"></path> <g opacity="0.5"> <path d="M17.0866 2.61039C16.2268 2.49997 15.1321 2.49998 13.7675 2.5H10.6778C9.31314 2.49998 8.21844 2.49997 7.35863 2.61039C6.46826 2.72473 5.72591 2.96835 5.13712 3.53075C4.79755 3.8551 4.56886 4.22833 4.41309 4.64928C4.91729 4.41928 5.48734 4.28374 6.12735 4.20084C7.21173 4.06037 8.5973 4.06038 10.3466 4.06039H14.2615C16.0108 4.06038 17.3963 4.06037 18.4807 4.20084C19.0397 4.27325 19.5453 4.38581 20.0003 4.56638C19.8457 4.17917 19.6253 3.83365 19.3081 3.53075C18.7193 2.96835 17.977 2.72473 17.0866 2.61039Z" fill="currentColor"></path> </g> </g></svg>
              {!(isCollapsed && !isHovering) && <span>Image Generation</span>}
          </Link>

          <Link to="/knowledge-tuning" className="sidebar-button finetuning-button" title={isCollapsed ? 'Finetuning Knowledge Learning' : ''}>
          <svg viewBox="0 0 24 24" fill="none" className="sidebar-icon" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <path opacity="0.4" d="M21.4707 5V19C21.4707 20.66 20.1207 22 18.4707 22H14.4707C12.8107 22 11.4707 20.66 11.4707 19V5C11.4707 3.34 12.8107 2 14.4707 2H18.4707C20.1207 2 21.4707 3.34 21.4707 5Z" fill="currentColor"></path> <path d="M17.2207 6C17.2207 6.41 16.8807 6.75 16.4707 6.75H11.4707V5.25H16.4707C16.8807 5.25 17.2207 5.59 17.2207 6Z" fill="currentColor"></path> <path d="M16.1207 18C16.1207 18.41 15.7907 18.75 15.3707 18.75H11.4707V17.25H15.3707C15.7907 17.25 16.1207 17.59 16.1207 18Z" fill="currentColor"></path> <path d="M17.2207 14.0102C17.2107 14.4202 16.8807 14.7502 16.4707 14.7502C16.4607 14.7502 16.4607 14.7502 16.4607 14.7502L11.4707 14.7002V13.2002L16.4707 13.2502C16.8907 13.2502 17.2207 13.5902 17.2207 14.0102Z" fill="currentColor"></path> <path d="M15.0307 10C15.0307 10.41 14.6907 10.75 14.2807 10.75H11.4707V9.25H14.2807C14.6907 9.25 15.0307 9.59 15.0307 10Z" fill="currentColor"></path> <path opacity="0.4" d="M8.4393 4.95V17.91C8.4393 18.36 8.2493 19.05 8.0193 19.43L7.1993 20.79C6.2593 22.37 4.7193 22.37 3.7693 20.79L2.9593 19.43C2.7193 19.05 2.5293 18.36 2.5293 17.91V4.95C2.5293 3.33 3.8593 2 5.4893 2C7.1093 2 8.4393 3.33 8.4393 4.95Z" fill="currentColor"></path> <path d="M8.4393 6.25H2.5293V7.75H8.4393V6.25Z" fill="currentColor"></path> </g></svg>
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
          <svg viewBox="0 0 24 24" fill="none" className="sidebar-icon" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <path opacity="0.4" d="M18.9401 5.41994L13.7701 2.42994C12.7801 1.85994 11.2301 1.85994 10.2401 2.42994L5.02008 5.43994C2.95008 6.83994 2.83008 7.04994 2.83008 9.27994V14.7099C2.83008 16.9399 2.95008 17.1599 5.06008 18.5799L10.2301 21.5699C10.7301 21.8599 11.3701 21.9999 12.0001 21.9999C12.6301 21.9999 13.2701 21.8599 13.7601 21.5699L18.9801 18.5599C21.0501 17.1599 21.1701 16.9499 21.1701 14.7199V9.27994C21.1701 7.04994 21.0501 6.83994 18.9401 5.41994Z" fill="currentColor"></path> <path d="M12 15.25C13.7949 15.25 15.25 13.7949 15.25 12C15.25 10.2051 13.7949 8.75 12 8.75C10.2051 8.75 8.75 10.2051 8.75 12C8.75 13.7949 10.2051 15.25 12 15.25Z" fill="currentColor"></path> </g></svg>
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