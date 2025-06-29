// externalAnalyzer/frontend-face-zephyrine/src/components/ChatFeed.jsx
import React, { useEffect, useState, useRef, memo, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import PropTypes from 'prop-types';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Copy, RefreshCw, Edit3 as Edit, Check, CheckCheck, X } from "lucide-react";
import rehypeRaw from 'rehype-raw';
import ErrorBoundary from './ErrorBoundary';
import { Virtuoso } from 'react-virtuoso'; // Replaced react-window with react-virtuoso

import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkMermaid from 'remark-mermaidjs';

// TypingIndicator Component with a more bubbly animation
const TypingIndicator = () => (
  <div className="typing-indicator">
    <div className="dot"></div>
    <div className="dot"></div>
    <div className="dot"></div>
  </div>
);

const MessageRow = memo(({ data, index, style }) => {
  const { messages, assistantName, onCopy, onRegenerate, onEditSave, copySuccessId, isGenerating, editingMessageId, editedContent, setEditedContent, handleSaveEdit, handleCancelEdit } = data;
  const message = messages[index];
  const prevMessage = messages[index - 1];
  const nextMessage = messages[index + 1];

  const isFirstInGroup = !prevMessage || prevMessage.sender !== message.sender;
  const isLastInGroup = !nextMessage || nextMessage.sender !== message.sender;

  let messageGroupClass = '';
  if (isFirstInGroup && isLastInGroup) messageGroupClass = 'group-single';
  else if (isFirstInGroup) messageGroupClass = 'group-start';
  else if (isLastInGroup) messageGroupClass = 'group-end';
  else messageGroupClass = 'group-middle';

  // This is the crucial fix. We apply flexbox alignment directly to the row's style.
  const rowStyle = {
      ...style,
      display: 'flex',
      justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start',
  };

  return (
      <div style={rowStyle}>
          <li className={`message ${message.sender} ${messageGroupClass}`}>
              <div className="message-avatar-wrapper">
                  {isLastInGroup && (
                      <div className="message-avatar">
                          {message.sender === 'user' ? 'U' : <img src="/img/AdelaideEntity.png" alt="Assistant Avatar" className="avatar-image" />}
                      </div>
                  )}
              </div>
              <div className="message-content-container">
                  {isFirstInGroup && message.sender === 'assistant' && (
                      <div className="message-sender-name">{assistantName}</div>
                  )}
                  <div className="message-content">{data.renderMessageContent(message.content)}</div>
                   <div className="message-actions">
                      {/* Action buttons can be rendered here */}
                  </div>
              </div>
          </li>
      </div>
  );
});
MessageRow.displayName = 'MessageRow';

const ChatFeed = ({
  messages,
  streamingMessage,
  isGenerating,
  assistantName,
  onCopy,
  onRegenerate,
  onEditSave,
  copySuccessId,
  lastAssistantMessageIndex,
}) => {
  const [editingMessageId, setEditingMessageId] = useState(null);
  const [editedContent, setEditedContent] = useState("");
  const virtuosoRef = useRef(null);

  const processMessageContent = (content) => {
    if (typeof content !== 'string') return '';
    const thinkBlockRegex = /<think>([\s\S]*?)<\/think>/gi;
    return content.replace(thinkBlockRegex, (match, thinkContent) => {
      const trimmedThinkContent = thinkContent.trim();
      const escapedThinkContent = trimmedThinkContent
        .replace(/&/g, "&")
        .replace(/</g, "<")
        .replace(/>/g, ">")
        .replace(/'/g, "'");
      return `<details class="thought-block">
                <summary class="thought-summary"><span class="summary-icon"></span>🧠</summary>
                <div class="thought-content">${escapedThinkContent}</div>
              </details>`;
    });
  };

  useEffect(() => {
    if (virtuosoRef.current) {
        virtuosoRef.current.scrollToIndex({
            index: messages.length - 1,
            align: 'end',
            behavior: 'smooth',
        });
    }
  }, [messages, streamingMessage]);

  const handleEditClick = useCallback((message) => {
    setEditingMessageId(message.id);
    setEditedContent(message.content);
  }, []);

  const handleCancelEdit = useCallback(() => {
    setEditingMessageId(null);
    setEditedContent("");
  }, []);

  const handleSaveEdit = useCallback(() => {
    if (editingMessageId && editedContent.trim()) {
      onEditSave(editingMessageId, editedContent);
      setEditingMessageId(null);
      setEditedContent("");
    } else {
      handleCancelEdit();
    }
  }, [editingMessageId, editedContent, onEditSave, handleCancelEdit]);

  const renderMessageContent = useCallback((contentToRender) => {
    return (
      <ErrorBoundary>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath, remarkMermaid]}
        rehypePlugins={[rehypeRaw, rehypeKatex]}
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
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
      >
        {processMessageContent(contentToRender)}
      </ReactMarkdown>
      </ErrorBoundary>
    );
  }, []);

  const Row = ({ index }) => {
    const message = messages[index];
    const prevMessage = messages[index - 1];
    const nextMessage = messages[index + 1];
    const editTextAreaRef = useRef(null);

    const isFirstInGroup = !prevMessage || prevMessage.sender !== message.sender;
    const isLastInGroup = !nextMessage || nextMessage.sender !== message.sender;

    let messageGroupClass = '';
    if (isFirstInGroup && isLastInGroup) messageGroupClass = 'group-single';
    else if (isFirstInGroup) messageGroupClass = 'group-start';
    else if (isLastInGroup) messageGroupClass = 'group-end';
    else messageGroupClass = 'group-middle';

    useEffect(() => {
        // Auto-resize textarea when editing
        if (editingMessageId === message.id && editTextAreaRef.current) {
            editTextAreaRef.current.style.height = "auto";
            editTextAreaRef.current.style.height = `${editTextAreaRef.current.scrollHeight}px`;
        }
    }, [editedContent, editingMessageId, message.id]);

    return (
        <li className={`message ${message.sender} ${messageGroupClass}`}>
            <div className="message-avatar-wrapper">
              {isLastInGroup && (
                <div className="message-avatar">
                  {message.sender === 'user' ? 'U' : <img src="/img/AdelaideEntity.png" alt="Assistant Avatar" className="avatar-image" />}
                </div>
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
                  {message.sender === 'assistant' && isFirstInGroup && (
                    <div className="message-sender-name">{assistantName}</div>
                  )}
                  <div className="message-content">
                    {renderMessageContent(message.content)}
                    {message.error && <span className="message-error"> ({message.error})</span>}
                  </div>
                  {message.sender === 'user' && (
                      <div className="message-status-icons">
                          {message.status === 'sending' && <Check size={14} className="status-sending" />}
                          {message.status === 'delivered' && <CheckCheck size={14} className="status-delivered" />}
                          {message.status === 'read' && <CheckCheck size={14} className="status-read" />}
                      </div>
                  )}
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
    );
  };

  const ListContainer = React.forwardRef(({ style, children }, ref) => (
    <ul ref={ref} style={style} className="message-list-virtualized">
        {children}
    </ul>
  ));
  ListContainer.displayName = "ListContainer";

  return (
    <div id="messages-container" className="message-list">
        <Virtuoso
            ref={virtuosoRef}
            style={{ height: '100%', width: '127%' }}
            data={messages}
            itemContent={(index) => <Row index={index} />}
            components={{ List: ListContainer }}
            followOutput="auto"
        />

        {/* The streaming message is rendered outside the virtualized list for simplicity */}
        {streamingMessage && (
            <ul className="message-list">
                <li
                key={streamingMessage.id || "streaming-message"}
                className="message assistant streaming group-single"
                >
                <div className="message-avatar-wrapper">
                    <div className="message-avatar">
                    <img
                        src="/img/AdelaideEntity.png"
                        alt="Assistant Avatar"
                        className="avatar-image"
                    />
                    </div>
                </div>
                <div className="message-content-container">
                    <div className="message-sender-name">{assistantName}</div>
                    {streamingMessage.isThinking ? (
                    <TypingIndicator />
                    ) : streamingMessage.content ? (
                    <div className="message-content">
                        <ReactMarkdown
                        remarkPlugins={[remarkMath, remarkMermaid]}
                        rehypePlugins={[rehypeRaw, rehypeKatex]}
                        children={processMessageContent(streamingMessage.content) + (isGenerating ? "<span class='streaming-cursor'>▋</span>" : "")}
                        components={{
                            code({ node, inline, className, children, ...props }) {
                            const match = /language-(\w+)/.exec(className || "");
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
                        {streamingMessage.tokensPerSecond && (
                        <span className="tokens-per-second">
                            {streamingMessage.tokensPerSecond} chars/s
                        </span>
                        )}
                    </div>
                    ) : null}
                </div>
                </li>
            </ul>
        )}
        <div id="bottom"></div>
    </div>
  );
};

ChatFeed.propTypes = {
  messages: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      sender: PropTypes.string.isRequired,
      content: PropTypes.string.isRequired,
      error: PropTypes.string,
      status: PropTypes.string,
    })
  ).isRequired,
  streamingMessage: PropTypes.shape({
    id: PropTypes.string,
    content: PropTypes.string,
    isLoading: PropTypes.bool,
    tokensPerSecond: PropTypes.string,
    isThinking: PropTypes.bool,
  }),
  isGenerating: PropTypes.bool,
  assistantName: PropTypes.string,
  onCopy: PropTypes.func.isRequired,
  onRegenerate: PropTypes.func.isRequired,
  onEditSave: PropTypes.func.isRequired,
  copySuccessId: PropTypes.string,
  lastAssistantMessageIndex: PropTypes.number,
};

ChatFeed.defaultProps = {
  streamingMessage: null,
  isGenerating: false,
  assistantName: "Zephyrine",
  copySuccessId: null,
  lastAssistantMessageIndex: -1,
};

export default memo(ChatFeed);