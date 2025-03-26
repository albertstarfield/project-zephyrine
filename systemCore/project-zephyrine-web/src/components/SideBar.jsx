import React, { useState } from "react"; // Import useState
import { useNavigate } from "react-router-dom";
import { v4 as uuidv4 } from "uuid";
import "../styles/components/_sidebar.css"; // Import sidebar styles

// Placeholder icons for collapse/expand and hamburger
const CollapseIcon = () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line></svg>;
const ExpandIcon = () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="15" y1="3" x2="15" y2="21"></line></svg>;
const HamburgerIcon = () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>;


// Receive isCollapsed and toggleSidebar as props
const SideBar = ({ systemInfo, isCollapsed, toggleSidebar }) => {
  const navigate = useNavigate();
  // State is now managed by App.jsx

  const handleNewChat = () => {
    navigate(`/chat/${uuidv4()}`);
  };

  // Placeholder for chat history items
  const chatHistory = [
    { id: "chat1", title: "React Basics Discussion" },
    { id: "chat2", title: "CSS Flexbox Help" },
    { id: "chat3", title: "JavaScript Function Example" },
  ];

  return (
    <>
      {/* Hamburger button for mobile - styling will handle visibility */}
      <button className="sidebar-hamburger-button" onClick={toggleSidebar} aria-label="Toggle sidebar">
        <HamburgerIcon />
      </button>

      <aside className={`sidebar ${isCollapsed ? "sidebar--collapsed" : ""}`}>
        {/* Desktop Collapse/Expand Button - Placed at the top for now */}
        <button className="sidebar-toggle-button sidebar-toggle-desktop" onClick={toggleSidebar} title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}>
          {isCollapsed ? <ExpandIcon /> : <CollapseIcon />}
        </button>

        <div className="sidebar-top-actions">
          {/* New Chat Button */}
          <button className="sidebar-button new-chat-button" onClick={handleNewChat} title={isCollapsed ? "New Chat" : ""}>
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
              <path d="M12 7V17M7 12H17" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"></path>
              <path opacity="0.5" d="M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z" strokeWidth="2"></path>
            </g>
          </svg>
          {!isCollapsed && <span>New Chat</span>} {/* Conditionally render text */}
        </button>
        {/* Placeholder for Model Selector */}
        {!isCollapsed && ( // Hide model selector when collapsed
          <div className="sidebar-button model-selector-placeholder">
            <span>GPT-4o</span> {/* Default/Current model */}
            {/* Dropdown icon/arrow */}
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="dropdown-arrow"><g id="SVGRepo_bgCarrier" strokeWidth="0"></g><g id="SVGRepo_tracerCarrier" strokeLinecap="round" strokeLinejoin="round"></g><g id="SVGRepo_iconCarrier"> <path d="M7 10L12 15L17 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path> </g></svg>
          </div>
        )}
      </div>

      {/* Chat History Section */}
      <nav className="sidebar-history">
        {!isCollapsed && <h4>History</h4>} {/* Hide title when collapsed */}
        <ul>
          {chatHistory.map(chat => (
            <li key={chat.id} title={isCollapsed ? chat.title : ""}> {/* Add title for tooltip when collapsed */}
              <button onClick={() => navigate(`/chat/${chat.id}`)}>
                {/* Placeholder for potential icon */}
                {!isCollapsed && chat.title} {/* Hide text when collapsed */}
              </button>
              {/* Add rename/delete icons here later */}
            </li>
          ))}
        </ul>
      </nav>

      {/* GPTs Section Placeholder */}
      <nav className="sidebar-gpts">
        {!isCollapsed && <h4>GPTs</h4>} {/* Hide title when collapsed */}
         {/* Placeholder for recently used/pinned GPTs */}
        <ul>
          <li title={isCollapsed ? "Web Browser" : ""}>
            <button className="sidebar-button">
              <span>🌐</span> {/* Keep icon */}
              {!isCollapsed && <span>&nbsp;Web Browser</span>} {/* Hide text */}
            </button>
          </li>
          <li title={isCollapsed ? "DALL-E" : ""}>
            <button className="sidebar-button">
              <span>🎨</span> {/* Keep icon */}
              {!isCollapsed && <span>&nbsp;DALL-E</span>} {/* Hide text */}
            </button>
          </li>
        </ul>
      </nav>

      {/* Projects Section Placeholder */}
      <nav className="sidebar-projects">
         {/* Could be a button or a list */}
         <button className="sidebar-button" title={isCollapsed ? "Projects" : ""}>
           {/* Placeholder Icon */}
           <svg viewBox="0 0 24 24" className="sidebar-icon" fill="none" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" strokeWidth="0"></g><g id="SVGRepo_tracerCarrier" strokeLinecap="round" strokeLinejoin="round"></g><g id="SVGRepo_iconCarrier"> <path d="M9 4H15M9 4C8.44772 4 8 4.44772 8 5V7C8 7.55228 8.44772 8 9 8H15C15.5523 8 16 7.55228 16 7V5C16 4.44772 15.5523 4 15 4M9 4C6.52166 4 4.68603 4.44384 3.50389 5.28131C2.32175 6.11878 2 7.1433 2 9.19234V14.8077C2 16.8567 2.32175 17.8812 3.50389 18.7187C4.68603 19.5562 6.52166 20 9 20H15C17.4783 20 19.314 19.5562 20.4961 18.7187C21.6782 17.8812 22 16.8567 22 14.8077V9.19234C22 7.1433 21.6782 6.11878 20.4961 5.28131C19.314 4.44384 17.4783 4 15 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"></path> <path d="M12 11V17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"></path> <path d="M9 14L12 17L15 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"></path> </g></svg>
           {!isCollapsed && <span>Projects</span>} {/* Hide text */}
         </button>
      </nav>

      {/* Spacer to push bottom content down */}
      <div className="sidebar-spacer"></div>

      {/* Bottom Actions (User, Settings) */}
      <div className="sidebar-bottom-actions">
         {/* User Button */}
        <button className="sidebar-button" title={isCollapsed ? (systemInfo.username || "User") : ""}>
          <svg viewBox="0 0 24 24" className="sidebar-icon" fill="none" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" strokeWidth="0"></g><g id="SVGRepo_tracerCarrier" strokeLinecap="round" strokeLinejoin="round"></g><g id="SVGRepo_iconCarrier"> <path opacity="0.5" d="M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z" stroke="currentColor" strokeWidth="1.5"></path> <path d="M12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15Z" stroke="currentColor" strokeWidth="1.5"></path> </g></svg>
          {!isCollapsed && <span>{systemInfo.username || "User"}</span>} {/* Hide text */}
        </button>
         {/* Settings Button Placeholder */}
         <button className="sidebar-button" title={isCollapsed ? "Settings" : ""}>
           <svg viewBox="0 0 24 24" className="sidebar-icon" fill="none" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" strokeWidth="0"></g><g id="SVGRepo_tracerCarrier" strokeLinecap="round" strokeLinejoin="round"></g><g id="SVGRepo_iconCarrier"> <path d="M16.694 9.506L15.3808 8.1928C14.818 7.63001 14.818 6.70968 15.3808 6.14689L16.694 4.83369C17.2568 4.2709 18.1771 4.2709 18.7399 4.83369L20.0531 6.14689C20.6159 6.70968 20.6159 7.63001 20.0531 8.1928L18.7399 9.506C18.1771 10.0688 17.2568 10.0688 16.694 9.506Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path> <path d="M16.694 19.1663L15.3808 17.8531C14.818 17.2903 14.818 16.3699 15.3808 15.8071L16.694 14.4939C17.2568 13.9311 18.1771 13.9311 18.7399 14.4939L20.0531 15.8071C20.6159 16.3699 20.6159 17.2903 20.0531 17.8531L18.7399 19.1663C18.1771 19.7291 17.2568 19.7291 16.694 19.1663Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path> <path d="M8.1928 9.506L6.8796 8.1928C6.31681 7.63001 6.31681 6.70968 6.8796 6.14689L8.1928 4.83369C8.75559 4.2709 9.67592 4.2709 10.2387 4.83369L11.5519 6.14689C12.1147 6.70968 12.1147 7.63001 11.5519 8.1928L10.2387 9.506C9.67592 10.0688 8.75559 10.0688 8.1928 9.506Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path> <path d="M8.1928 19.1663L6.8796 17.8531C6.31681 17.2903 6.31681 16.3699 6.8796 15.8071L8.1928 14.4939C8.75559 13.9311 9.67592 13.9311 10.2387 14.4939L11.5519 15.8071C12.1147 16.3699 12.1147 17.2903 11.5519 17.8531L10.2387 19.1663C9.67592 19.7291 8.75559 19.7291 8.1928 19.1663Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"></path> </g></svg>
           {!isCollapsed && <span>Settings</span>} {/* Hide text */}
         </button>
      </div>
    </aside>
    </> // Close the fragment
  );
};

export default SideBar;
