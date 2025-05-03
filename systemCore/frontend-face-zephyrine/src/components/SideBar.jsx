import React, { useState, useEffect, useRef } from 'react'; // Import useState, useEffect, useRef
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext'; // Import useAuth
import '../styles/components/_sidebar.css';

// --- Icons ---
const CollapseIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
    <line x1="9" y1="3" x2="9" y2="21"></line>
  </svg>
);
const ExpandIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
    <line x1="15" y1="3" x2="15" y2="21"></line>
  </svg>
);
const HamburgerIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <line x1="3" y1="12" x2="21" y2="12"></line>
    <line x1="3" y1="6" x2="21" y2="6"></line>
    <line x1="3" y1="18" x2="21" y2="18"></line>
  </svg>
);

// Placeholder Icons for Edit/Delete
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

const CheckIcon = () => ( // Icon to confirm rename
  <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16" className="sidebar-action-icon confirm-icon">
      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"></path>
  </svg>
);

const CancelIcon = () => ( // Icon to cancel rename
    <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16" className="sidebar-action-icon cancel-icon">
        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd"></path>
    </svg>
);
// --- End Icons ---


// Receive user, onNewChat, chatHistory, onRenameChat, onDeleteChat props
const SideBar = ({
  systemInfo,
  isCollapsed,
  toggleSidebar,
  user,
  onNewChat,
  chatHistory,
  onRenameChat,
  onDeleteChat,

}) => {
  const navigate = useNavigate();
  const { signOut } = useAuth();
  const [isUserMenuExpanded, setIsUserMenuExpanded] = useState(true);
  const [editingChatId, setEditingChatId] = useState(null); // Track which chat is being edited
  const [editText, setEditText] = useState(''); // Store the temporary edit text
  const inputRef = useRef(null); // Ref for the input field

  // Focus input when editing starts
  useEffect(() => {
    if (editingChatId && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select(); // Select text for easy replacement
    }
  }, [editingChatId]);


  const handleNewChatClick = () => {
    if (onNewChat) {
      onNewChat();
    }
  };

  const toggleUserMenu = () => {
    setIsUserMenuExpanded(!isUserMenuExpanded);
  };

  const handleLogout = async () => {
    await signOut();
  };

  // --- Edit/Delete Handlers ---
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
      onRenameChat(chatId, editText.trim()); // Call prop function
    }
    cancelEditing(); // Exit editing mode
  };

  const handleDelete = (chatId) => {
    // Optional: Add confirmation dialog here
    if (onDeleteChat) {
      onDeleteChat(chatId); // Call prop function
    }
     // If the currently viewed chat is deleted, navigate away (e.g., to home or new chat)
    // This logic might be better handled in the parent component after deletion
  };

  // Handle Enter key press in input
  const handleInputKeyDown = (event, chatId) => {
    if (event.key === 'Enter') {
      handleRename(chatId);
    } else if (event.key === 'Escape') {
      cancelEditing();
    }
  };

  // Handle input blur (losing focus)
  const handleInputBlur = (chatId) => {
    // Delay slightly to allow confirm/cancel icon clicks
    setTimeout(() => {
        // Check if still editing the same item before saving on blur
        if (editingChatId === chatId) {
             handleRename(chatId);
        }
    }, 100); // 100ms delay
  };


  return (
    <>
      {/* Hamburger button for mobile - styling will handle visibility */}
      <button
        className="sidebar-hamburger-button"
        onClick={toggleSidebar}
        aria-label="Toggle sidebar"
      >
        <HamburgerIcon />
      </button>

      <aside className={`sidebar ${isCollapsed ? "sidebar--collapsed" : ""}`}>
        {/* Desktop Collapse/Expand Button - Placed at the top for now */}
        <button
          className="sidebar-toggle-button sidebar-toggle-desktop"
          onClick={toggleSidebar}
          title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isCollapsed ? <ExpandIcon /> : <CollapseIcon />}
        </button>

        <div className="sidebar-top-actions">
          {/* New Chat Button - Use handleNewChatClick */}
          <button
            className="sidebar-button new-chat-button"
            onClick={handleNewChatClick} // Use the passed handler
            title={isCollapsed ? 'New Chat' : ''}
          >
            <svg
              viewBox="0 0 24 24"
              className="sidebar-icon"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              stroke="currentColor"
            >
              <g id="SVGRepo_bgCarrier" strokeWidth="0"></g>
              <g
                id="SVGRepo_tracerCarrier"
                strokeLinecap="round"
                strokeLinejoin="round"
              ></g>
              <g id="SVGRepo_iconCarrier">
                <path
                  d="M12 7V17M7 12H17"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                ></path>
                <path
                  opacity="0.5"
                  d="M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z"
                  strokeWidth="2"
                ></path>
              </g>
            </svg>
            {!isCollapsed && <span>New Chat</span>}{" "}
            {/* Conditionally render text */}
          </button>
          {/* Model Selector Dropdown */}
          {!isCollapsed && (
            <div className="model-selector-container">
              {/* Custom arrow overlay */}
              <svg
                className="dropdown-arrow"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                 <path
                    d="M7 10L12 15L17 10"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  ></path>
              </svg>
            </div>
          )}
        </div>

        {/* Chat History Section */}
        <nav className="sidebar-history">
          {!isCollapsed && <h4>History</h4>} {/* Hide title when collapsed */}
          <ul>
          {(chatHistory || []).map(chat => (
              <li
                key={chat.id}
                title={isCollapsed ? chat.title : ""}
                className={`history-item ${editingChatId === chat.id ? 'editing' : ''}`}
              >
                {editingChatId === chat.id ? (
                  // --- Editing State ---
                  <div className="history-item-edit-container">
                    <input
                      ref={inputRef}
                      type="text"
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      onKeyDown={(e) => handleInputKeyDown(e, chat.id)}
                      onBlur={() => handleInputBlur(chat.id)}
                      className="history-item-input"
                    />
                    <div className="history-item-edit-actions">
                       <button onClick={() => handleRename(chat.id)} className="icon-button" title="Confirm rename">
                         <CheckIcon />
                       </button>
                       <button onClick={cancelEditing} className="icon-button" title="Cancel rename">
                         <CancelIcon />
                       </button>
                    </div>
                  </div>
                ) : (
                  // --- Normal State ---
                  <button
                    className="history-item-button" // Use a more specific class
                    onClick={() => !isCollapsed && navigate(`/chat/${chat.id}`)} // Prevent nav when collapsed? Or handle differently?
                  >
                    <span className="history-item-title">
                      {!isCollapsed && chat.title}
                    </span>
                    {!isCollapsed && ( // Show icons only when not collapsed and not editing
                      <div className="history-item-actions">
                        <button onClick={(e) => { e.stopPropagation(); startEditing(chat); }} className="icon-button" title="Rename chat">
                          <EditIcon />
                        </button>
                        <button onClick={(e) => { e.stopPropagation(); handleDelete(chat.id); }} className="icon-button" title="Delete chat">
                          <DeleteIcon />
                        </button>
                      </div>
                    )}
                  </button>
                )}
              </li>
            ))}
          </ul>
        </nav>

        {/* Conditionally render GPTs and Projects based on isUserMenuExpanded and !isCollapsed */}
        {!isCollapsed && isUserMenuExpanded && (
          <>
            {/* GPTs Section Placeholder */}
            <nav className="sidebar-gpts">
              <h4>GDTs</h4> {/* Title only shown when expanded */}
              {/* Placeholder for recently used/pinned GPTs */}
              <ul>
                <li> {/* Removed title as it's not needed when expanded */}
                  <button className="sidebar-button">
                    <span>🌐</span> {/* Keep icon */}
                    <span>&nbsp;Web Browser</span> {/* Show text */}
                  </button>
                </li>
                <li> {/* Removed title */}
                  <button className="sidebar-button">
                    <span>🎨</span> {/* Keep icon */}
                    <span>&nbsp;Noisy thinking</span> {/* Show text */}
                  </button>
                </li>
              </ul>
            </nav>

            {/* Projects Section Placeholder */}
            <nav className="sidebar-projects">
              {/* Could be a button or a list */}
              <button
                className="sidebar-button"
                // Removed title
              >
                {/* Placeholder Icon */}
                <svg
                  viewBox="0 0 24 24"
                  className="sidebar-icon"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <g id="SVGRepo_bgCarrier" strokeWidth="0"></g>
                  <g
                    id="SVGRepo_tracerCarrier"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  ></g>
                  <g id="SVGRepo_iconCarrier">
                    {" "}
                    <path
                      d="M9 4H15M9 4C8.44772 4 8 4.44772 8 5V7C8 7.55228 8.44772 8 9 8H15C15.5523 8 16 7.55228 16 7V5C16 4.44772 15.5523 4 15 4M9 4C6.52166 4 4.68603 4.44384 3.50389 5.28131C2.32175 6.11878 2 7.1433 2 9.19234V14.8077C2 16.8567 2.32175 17.8812 3.50389 18.7187C4.68603 19.5562 6.52166 20 9 20H15C17.4783 20 19.314 19.5562 20.4961 18.7187C21.6782 17.8812 22 16.8567 22 14.8077V9.19234C22 7.1433 21.6782 6.11878 20.4961 5.28131C19.314 4.44384 17.4783 4 15 4"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                    ></path>{" "}
                    <path
                      d="M12 11V17"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                    ></path>{" "}
                    <path
                      d="M9 14L12 17L15 14"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                    ></path>{" "}
                  </g>
                </svg>
                <span>Projects</span> {/* Show text */}
              </button>
            </nav>
          </>
        )}

        {/* Spacer to push bottom content down */}
        <div className="sidebar-spacer"></div>

        {/* Bottom Actions (User, Logout, Settings) */}
        <div className="sidebar-bottom-actions">
          {/* User Button - Display user email */}
          <button
            className="sidebar-button user-button" // Added user-button class for potential styling
            title={isCollapsed ? user?.email : (isUserMenuExpanded ? 'Collapse Menu' : 'Expand Menu')}
            onClick={toggleUserMenu} // Add onClick handler
          >
            <svg
              viewBox="0 0 24 24"
              className="sidebar-icon"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <g id="SVGRepo_bgCarrier" strokeWidth="0"></g>
              <g
                id="SVGRepo_tracerCarrier"
                strokeLinecap="round"
                strokeLinejoin="round"
              ></g>
              <g id="SVGRepo_iconCarrier">
                {" "}
                <path
                  opacity="0.5"
                  d="M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                ></path>{" "}
                <path
                  d="M12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15Z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                ></path>{" "}
              </g>
            </svg>
            {!isCollapsed && <span className="user-email">{user?.email}</span>}{' '}
            {/* Display email */}
          </button>
          {/* Logout Button */}
          <button
            className="sidebar-button logout-button" // Added logout-button class
            onClick={handleLogout}
            title={isCollapsed ? 'Logout' : ''}
          >
            {/* Simple Logout Icon */}
            <svg
              viewBox="0 0 24 24"
              className="sidebar-icon"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              stroke="currentColor"
            >
              <g id="SVGRepo_bgCarrier" strokeWidth="0"></g>
              <g id="SVGRepo_tracerCarrier" strokeLinecap="round" strokeLinejoin="round"></g>
              <g id="SVGRepo_iconCarrier">
                <path d="M15 3H7C5.89543 3 5 3.89543 5 5V19C5 20.1046 5.89543 21 7 21H15" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path>
                <path d="M19 12L15 8M19 12L15 16M19 12H9" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path>
              </g>
            </svg>
            {!isCollapsed && <span>Logout</span>}
          </button>
          {/* Settings Button Placeholder */}
          <button
            className="sidebar-button settings-button" // Added settings-button class
            title={isCollapsed ? 'Settings' : ''}
          >
            <svg
              viewBox="0 0 24 24"
              className="sidebar-icon"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <g id="SVGRepo_bgCarrier" strokeWidth="0"></g>
              <g
                id="SVGRepo_tracerCarrier"
                strokeLinecap="round"
                strokeLinejoin="round"
              ></g>
              <g id="SVGRepo_iconCarrier">
                {" "}
                <path
                  d="M16.694 9.506L15.3808 8.1928C14.818 7.63001 14.818 6.70968 15.3808 6.14689L16.694 4.83369C17.2568 4.2709 18.1771 4.2709 18.7399 4.83369L20.0531 6.14689C20.6159 6.70968 20.6159 7.63001 20.0531 8.1928L18.7399 9.506C18.1771 10.0688 17.2568 10.0688 16.694 9.506Z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                ></path>{" "}
                <path
                  d="M16.694 19.1663L15.3808 17.8531C14.818 17.2903 14.818 16.3699 15.3808 15.8071L16.694 14.4939C17.2568 13.9311 18.1771 13.9311 18.7399 14.4939L20.0531 15.8071C20.6159 16.3699 20.6159 17.2903 20.0531 17.8531L18.7399 19.1663C18.1771 19.7291 17.2568 19.7291 16.694 19.1663Z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                ></path>{" "}
                <path
                  d="M8.1928 9.506L6.8796 8.1928C6.31681 7.63001 6.31681 6.70968 6.8796 6.14689L8.1928 4.83369C8.75559 4.2709 9.67592 4.2709 10.2387 4.83369L11.5519 6.14689C12.1147 6.70968 12.1147 7.63001 11.5519 8.1928L10.2387 9.506C9.67592 10.0688 8.75559 10.0688 8.1928 9.506Z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                ></path>{" "}
                <path
                  d="M8.1928 19.1663L6.8796 17.8531C6.31681 17.2903 6.31681 16.3699 6.8796 15.8071L8.1928 14.4939C8.75559 13.9311 9.67592 13.9311 10.2387 14.4939L11.5519 15.8071C12.1147 16.3699 12.1147 17.2903 11.5519 17.8531L10.2387 19.1663C9.67592 19.7291 8.75559 19.7291 8.1928 19.1663Z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                ></path>{" "}
              </g>
            </svg>
            {!isCollapsed && <span>Settings</span>} {/* Hide text */}
          </button>
        </div>
      </aside>
    </> // Close the fragment
  );
};

export default SideBar;
