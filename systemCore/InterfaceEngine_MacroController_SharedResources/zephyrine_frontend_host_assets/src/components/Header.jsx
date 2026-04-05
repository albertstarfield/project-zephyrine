import React from "react";
import "../styles/Header.css";

const Header = ({ assistantName, username }) => {
  return (
    <header className="app-header">
      <div className="logo">
        <span className="logo-text">Project {assistantName}</span>
      </div>
      <div className="header-controls">
        <span className="username">Welcome, {username}</span>
      </div>
    </header>
  );
};

export default Header;
