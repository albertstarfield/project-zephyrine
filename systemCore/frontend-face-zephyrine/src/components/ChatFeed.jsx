// externalAnalyzer/frontend-face-zephyrine/src/components/ChatFeed.jsx
import React, { useEffect, useState, useRef, memo } from "react";
import ReactMarkdown from "react-markdown";
import PropTypes from 'prop-types';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Copy, RefreshCw, Edit3 as Edit, Check, X } from "lucide-react";
import rehypeRaw from 'rehype-raw';
import ErrorBoundary from './ErrorBoundary'; // 1. Import the ErrorBoundary


// New Imports for Math and Mermaid
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm'; 
import remarkMermaid from 'remark-mermaidjs'; // Ensure you installed remark-mermaidjs correctly

// Note: You might need to initialize Mermaid directly if remark-mermaidjs doesn't
// automatically render. For now, it will simply render a <div>.
// For direct rendering, you might wrap the Mermaid div in a useEffect with mermaid.init()
// or use a dedicated component like 'react-mermaid'.
// This example assumes remark-mermaidjs handles the basic wrapping.


const ChatFeed = ({
  messages,
  streamingMessage,
  showPlaceholder,
  isGenerating,
  onExampleClick,
  bottomRef,
  assistantName,
  onCopy,
  onRegenerate,
  onEditSave,
  copySuccessId,
  lastAssistantMessageIndex,
}) => {
  const [fadeIn, setFadeIn] = useState(false);
  const [editingMessageId, setEditingMessageId] = useState(null);
  const [editedContent, setEditedContent] = useState("");
  const editTextAreaRef = useRef(null);

  // Function to pre-process message content for <think> blocks
  const processMessageContent = (content) => {
    if (typeof content !== 'string') {
      return '';
    }
    const thinkBlockRegex = /<think>([\s\S]*?)<\/think>/gi;
    return content.replace(thinkBlockRegex, (match, thinkContent) => {
      const trimmedThinkContent = thinkContent.trim();
      // Basic HTML escaping for the content within <details>
      const escapedThinkContent = trimmedThinkContent
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
      return `<details class="thought-block">
                <summary class="thought-summary"><span class="summary-icon"></span>View Zephy Backend Thoughts</summary>
                <div class="thought-content">${escapedThinkContent}</div>
              </details>`;
    });
  };

  useEffect(() => {
    if (messages.length > 0 || (streamingMessage && streamingMessage.content)) {
      setTimeout(() => {
        setFadeIn(true);
      }, 100);
    } else {
      setFadeIn(false);
    }
  }, [messages, streamingMessage]);

  const handleEditClick = (message) => {
    setEditingMessageId(message.id);
    setEditedContent(message.content);
  };

  const handleCancelEdit = () => {
    setEditingMessageId(null);
    setEditedContent("");
  };

  const handleSaveEdit = () => {
    if (editingMessageId && editedContent.trim()) {
      onEditSave(editingMessageId, editedContent);
      setEditingMessageId(null);
      setEditedContent("");
    } else {
      handleCancelEdit();
    }
  };

  useEffect(() => {
    if (editingMessageId && editTextAreaRef.current) {
      editTextAreaRef.current.style.height = "auto";
      editTextAreaRef.current.style.height = `${editTextAreaRef.current.scrollHeight}px`;
      editTextAreaRef.current.focus();
      editTextAreaRef.current.select();
    }
  }, [editedContent, editingMessageId]);


  const renderMessageContent = (contentToRender) => {
    return (
      <ErrorBoundary>
      <ReactMarkdown
        // Add remark plugins for math and mermaid
        remarkPlugins={[remarkGfm, remarkMath, remarkMermaid]}
        // Add rehype plugins for raw HTML (for <think> blocks) and KaTeX
        rehypePlugins={[rehypeRaw, rehypeKatex]}
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            // Handle Mermaid code blocks specifically
            if (className === 'language-mermaid') {
              // The 'mermaid' class is used by remark-mermaidjs.
              // You might need a global 'mermaid.init()' call or a custom component
              // to fully render these dynamically if remark-mermaidjs doesn't
              // handle it automatically. For now, it outputs a div with the code.
              return <div className="mermaid">{String(children).replace(/\n$/, "")}</div>;
            }
            return !inline && match ? (
              <SyntaxHighlighter
                style={atomDark}
                language={match[1]}
                PreTag="div"
                {...props}
              >
                {String(children).replace(/\n$/, "")}
              </SyntaxHighlighter>
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
          // For math, remark-math and rehype-katex usually handle it automatically
          // without needing specific component overrides for inline or block math.
        }}
      >
        {processMessageContent(contentToRender)}
      </ReactMarkdown>
      </ErrorBoundary>
    );
  };


  return (
    <>
      {showPlaceholder ? (
        <div id="feed-placeholder-alpaca">
          <div className="logo-container">
            <img
              src="/img/ProjectZephy023LogoRenewal.png"
              className="logo"
              alt="Project Zephyrine Logo"
            />
          </div>
          <div className="character-container">
            <img
              src="/img/AdelaideEntity.png"
              className="character-image"
              alt="Adelaide Entity"
            />
          </div>
          <div className="greeting-text">
            <p>
              Greetings! I am {assistantName || "Zephyrine"}. Delighted to make your
              acquaintance and wishing you a splendid day ahead. While I may
              possess greater capabilities, please bear in mind that my role is
              to assist you. You remain the driving force, the orchestrator of
              your own journey. Shall we embark on our interaction together?
            </p>
          </div>
          <div className="scroll-indicator">
            <svg
              viewBox="0 0 24 24"
              stroke="currentColor"
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
                <path
                  d="M19 9L14 14.1599C13.7429 14.4323 13.4329 14.6493 13.089 14.7976C12.7451 14.9459 12.3745 15.0225 12 15.0225C11.6255 15.0225 11.2549 14.9459 10.9109 14.7976C10.567 14.6493 10.2571 14.4323 10 14.1599L5 9"
                  stroke="#595959"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                ></path>
              </g>
            </svg>
            <p>Scroll down for usage examples</p>
          </div>
          <div className="example-categories">
            <div className="category">
              <div className="category-icon sun">
                <svg viewBox="0 0 24 24" stroke="currentColor" fill="none" xmlns="http://www.w3.org/2000/svg"><g><path fillRule="evenodd" clipRule="evenodd" d="M12.0002 1.25C12.4144 1.25 12.7502 1.58579 12.7502 2V3C12.7502 3.41421 12.4144 3.75 12.0002 3.75C11.586 3.75 11.2502 3.41421 11.2502 3V2C11.2502 1.58579 11.586 1.25 12.0002 1.25ZM1.25017 12C1.25017 11.5858 1.58596 11.25 2.00017 11.25H3.00017C3.41438 11.25 3.75017 11.5858 3.75017 12C3.75017 12.4142 3.41438 12.75 3.00017 12.75H2.00017C1.58596 12.75 1.25017 12.4142 1.25017 12ZM20.2502 12C20.2502 11.5858 20.586 11.25 21.0002 11.25H22.0002C22.4144 11.25 22.7502 11.5858 22.7502 12C22.7502 12.4142 22.4144 12.75 22.0002 12.75H21.0002C20.586 12.75 20.2502 12.4142 20.2502 12Z" fill="#f4a61f"></path><path d="M22.1722 16.0424C20.8488 15.7306 20.1708 14.9891 19.5458 14.0435C19.137 13.425 18.4905 13.1875 17.8692 13.2526C17.955 12.8486 18.0002 12.4296 18.0002 12C18.0002 8.68629 15.3139 6 12.0002 6C8.68646 6 6.00017 8.68629 6.00017 12C6.00017 12.3623 6.03228 12.7171 6.0938 13.0617C5.51698 13.0396 4.93367 13.2972 4.57004 13.8657C3.90775 14.9011 3.2291 15.7123 1.82816 16.0424C1.42498 16.1374 1.17516 16.5412 1.27016 16.9444C1.36516 17.3476 1.76901 17.5974 2.17218 17.5024C4.13929 17.0389 5.09416 15.8301 5.83365 14.674C5.88442 14.5946 5.95388 14.5619 6.02981 14.5602C6.11032 14.5585 6.20392 14.5934 6.27565 14.6853C7.38876 16.1101 9.15759 17.75 12.0002 17.75C14.7604 17.75 16.5386 16.4804 17.7063 14.9195C17.7936 14.8027 17.9223 14.7449 18.0408 14.7427C18.1519 14.7406 18.2374 14.7844 18.2944 14.8705C19.0206 15.9692 19.9762 17.066 21.8282 17.5024C22.2313 17.5974 22.6352 17.3478 22.7302 16.9446C22.8252 16.5414 22.5754 16.1376 22.1722 16.0426Z" fill="#f4a61f"></path><g opacity="0.5"><path d="M4.39936 4.39838C4.69225 4.10549 5.16712 4.10549 5.46002 4.39838L5.85285 4.79122C6.14575 5.08411 6.14575 5.55898 5.85285 5.85188C5.55996 6.14477 5.08509 6.14477 4.79219 5.85188L4.39936 5.45904C4.10646 5.16615 4.10646 4.69127 4.39936 4.39838Z" fill="#f4a61f"></path><path d="M19.6009 4.39864C19.8938 4.69153 19.8938 5.16641 19.6009 5.4593L19.2081 5.85214C18.9152 6.14503 18.4403 6.14503 18.1474 5.85214C17.8545 5.55924 17.8545 5.08437 18.1474 4.79148L18.5402 4.39864C18.8331 4.10575 19.308 4.10575 19.6009 4.39864Z" fill="#f4a61f"></path></g><g opacity="0.5"><path d="M4.57004 18.8659C5.25688 17.7921 6.72749 17.8273 7.4577 18.762C8.4477 20.0293 9.82965 21.2502 12.0002 21.2502C14.2088 21.2502 15.5699 20.2714 16.5051 19.0211C17.2251 18.0587 18.7909 17.9015 19.5458 19.0437C20.1708 19.9893 20.8488 20.7308 22.1722 21.0426C22.5754 21.1376 22.8252 21.5414 22.7302 21.9446C22.6352 22.3478 22.2313 22.5976 21.8282 22.5026C19.9762 22.0662 19.0206 20.9694 18.2944 19.8707C18.2374 19.7846 18.1519 19.7408 18.0408 19.7429C17.9223 19.7451 17.7936 19.8029 17.7063 19.9197C16.5386 21.4806 14.7604 22.7502 12.0002 22.7502C9.15759 22.7502 7.38876 21.1103 6.27565 19.6855C6.20392 19.5936 6.11032 19.5587 6.02981 19.5604C5.95388 19.5621 5.88442 19.5948 5.83365 19.6742C5.09416 20.8303 4.13929 22.0391 2.17218 22.5026C1.76901 22.5976 1.36516 22.3478 1.27016 21.9446C1.17516 21.5414 1.42498 21.1376 1.82816 21.0426C3.2291 20.7125 3.90775 19.9013 4.57004 18.8659Z" fill="#f4a61f"></path></g></g></svg>
              </div>
              <h3>Plan Vacation</h3>
              <div className="example-card">
                <button
                  onClick={() =>
                    onExampleClick && onExampleClick(
                      "Hi there, I want to have a relaxing summer vacation here. Can you make a list of recommended places where I should go?"
                    )
                  }
                >
                  "Hi there, I want to have a relaxing summer vacation here. Can
                  you make a list of recommended places where I should go?"
                </button>
              </div>
            </div>
            {/* ... other categories from your example ... */}
          </div>
        </div>
      ) : (
        <ul id="messages" className={`message-list ${fadeIn ? "fade-in" : ""}`}>
          {messages.map((message, index) => (
            <li
              key={message.id}
              className={`message ${
                message.sender === "user" ? "user" : "assistant"
              }`}
            >
              <div className="message-avatar">
                {message.sender === "user" ? (
                  "U"
                ) : (
                  <img
                    src="/img/AdelaideEntity.png"
                    alt="Assistant Avatar"
                    className="avatar-image"
                  />
                )}
              </div>
              <div className="message-content-container">
                {editingMessageId === message.id ? (
                  <div className="message-edit-area">
                    <textarea
                      ref={editTextAreaRef}
                      value={editedContent}
                      onChange={(e) => setEditedContent(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleSaveEdit();
                        }
                        if (e.key === 'Escape') {
                          e.preventDefault();
                          handleCancelEdit();
                        }
                      }}
                      rows={1}
                      style={{ overflowY: 'hidden' }}
                    />
                    <div className="message-edit-actions">
                      <button
                        onClick={handleSaveEdit}
                        className="message-action-button"
                        title="Save changes (Enter)"
                      >
                        <Check size={16} />
                      </button>
                      <button
                        onClick={handleCancelEdit}
                        className="message-action-button"
                        title="Cancel edit (Esc)"
                      >
                        <X size={16} />
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="message-content">
                      {renderMessageContent(message.content)}
                      {message.error && <span className="message-error"> ({message.error})</span>}
                    </div>
                    <div className="message-actions">
                      {message.sender === 'user' && !isGenerating && (
                         <button
                           onClick={() => handleEditClick(message)}
                           className="message-action-button"
                           title="Edit message"
                         >
                           <Edit size={14} />
                         </button>
                       )}
                      <button
                        onClick={() => onCopy(message.content, message.id)}
                        className="message-action-button"
                        title="Copy message"
                      >
                        {copySuccessId === message.id ? 'Copied!' : <Copy size={14} />}
                      </button>
                      {message.sender === 'assistant' &&
                       index === lastAssistantMessageIndex &&
                       !isGenerating &&
                       (!streamingMessage || !streamingMessage.content) && (
                        <button
                          onClick={() => onRegenerate(message.id)}
                          className="message-action-button"
                          title="Regenerate response"
                          disabled={isGenerating}
                        >
                          <RefreshCw size={14} />
                        </button>
                      )}
                    </div>
                  </>
                )}
              </div>
            </li>
          ))}

          {streamingMessage && streamingMessage.content && (
            <li
              key={streamingMessage.id || "streaming-message"}
              className="message assistant streaming"
            >
              <div className="message-avatar">
                <img
                  src="/img/AdelaideEntity.png"
                  alt="Assistant Avatar"
                  className="avatar-image"
                />
              </div>
              <div className="message-content-container">
                <div className="message-content">
                  {/* Process streaming content and append cursor */}
                  <ReactMarkdown
                    // Add remark plugins for math and mermaid
                    remarkPlugins={[remarkMath, remarkMermaid]}
                    // Add rehype plugins for raw HTML (for <think> blocks) and KaTeX
                    rehypePlugins={[rehypeRaw, rehypeKatex]}
                    children={processMessageContent(streamingMessage.content) + "<span class='streaming-cursor'>▋</span>"}
                    components={{
                      code({ node, inline, className, children, ...props }) {
                        const match = /language-(\w+)/.exec(className || "");
                        // Handle Mermaid code blocks specifically
                        if (className === 'language-mermaid') {
                          return <div className="mermaid">{String(children).replace(/\n$/, "")}</div>;
                        }
                        return !inline && match ? (
                          <SyntaxHighlighter
                            style={atomDark}
                            language={match[1]}
                            PreTag="div"
                            {...props}
                          >
                            {String(children).replace(/\n$/, "")}
                          </SyntaxHighlighter>
                        ) : (
                          <code className={className} {...props}>
                            {children}
                          </code>
                        );
                      },
                    }}
                  />
                </div>
              </div>
            </li>
          )}
          <div id="bottom" ref={bottomRef}></div>
        </ul>
      )}
    </>
  );
};

ChatFeed.propTypes = {
  messages: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      sender: PropTypes.string.isRequired,
      content: PropTypes.string.isRequired,
      error: PropTypes.string,
    })
  ).isRequired,
  streamingMessage: PropTypes.shape({
    id: PropTypes.string,
    content: PropTypes.string,
    isLoading: PropTypes.bool,
  }),
  showPlaceholder: PropTypes.bool,
  isGenerating: PropTypes.bool,
  onExampleClick: PropTypes.func,
  bottomRef: PropTypes.oneOfType([
    PropTypes.func,
    PropTypes.shape({ current: PropTypes.instanceOf(Element) }),
  ]),
  assistantName: PropTypes.string,
  onCopy: PropTypes.func.isRequired,
  onRegenerate: PropTypes.func.isRequired,
  onEditSave: PropTypes.func.isRequired,
  copySuccessId: PropTypes.string,
  lastAssistantMessageIndex: PropTypes.number,
};

ChatFeed.defaultProps = {
  streamingMessage: null,
  showPlaceholder: false,
  isGenerating: false,
  onExampleClick: () => {},
  assistantName: "Zephyrine",
  copySuccessId: null,
  lastAssistantMessageIndex: -1,
};

export default ChatFeed;