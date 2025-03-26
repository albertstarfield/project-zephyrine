import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomDark } from "react-syntax-highlighter/dist/esm/styles/prism";

const ChatFeed = ({
  messages,
  showPlaceholder,
  isGenerating,
  onExampleClick,
  bottomRef,
  assistantName,
}) => {
  const [fadeIn, setFadeIn] = useState(false);

  useEffect(() => {
    if (messages.length > 0) {
      setTimeout(() => {
        setFadeIn(true);
      }, 100);
    }
  }, [messages]);

  // Function to render message content with markdown support
  const renderMessageContent = (content) => {
    return (
      <ReactMarkdown
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
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
        {content}
      </ReactMarkdown>
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
              Greetings! I am {assistantName}. Delighted to make your
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
                      fillRule="evenodd"
                      clipRule="evenodd"
                      d="M12.0002 1.25C12.4144 1.25 12.7502 1.58579 12.7502 2V3C12.7502 3.41421 12.4144 3.75 12.0002 3.75C11.586 3.75 11.2502 3.41421 11.2502 3V2C11.2502 1.58579 11.586 1.25 12.0002 1.25ZM1.25017 12C1.25017 11.5858 1.58596 11.25 2.00017 11.25H3.00017C3.41438 11.25 3.75017 11.5858 3.75017 12C3.75017 12.4142 3.41438 12.75 3.00017 12.75H2.00017C1.58596 12.75 1.25017 12.4142 1.25017 12ZM20.2502 12C20.2502 11.5858 20.586 11.25 21.0002 11.25H22.0002C22.4144 11.25 22.7502 11.5858 22.7502 12C22.7502 12.4142 22.4144 12.75 22.0002 12.75H21.0002C20.586 12.75 20.2502 12.4142 20.2502 12Z"
                      fill="#f4a61f"
                    ></path>
                    <path
                      d="M22.1722 16.0424C20.8488 15.7306 20.1708 14.9891 19.5458 14.0435C19.137 13.425 18.4905 13.1875 17.8692 13.2526C17.955 12.8486 18.0002 12.4296 18.0002 12C18.0002 8.68629 15.3139 6 12.0002 6C8.68646 6 6.00017 8.68629 6.00017 12C6.00017 12.3623 6.03228 12.7171 6.0938 13.0617C5.51698 13.0396 4.93367 13.2972 4.57004 13.8657C3.90775 14.9011 3.2291 15.7123 1.82816 16.0424C1.42498 16.1374 1.17516 16.5412 1.27016 16.9444C1.36516 17.3476 1.76901 17.5974 2.17218 17.5024C4.13929 17.0389 5.09416 15.8301 5.83365 14.674C5.88442 14.5946 5.95388 14.5619 6.02981 14.5602C6.11032 14.5585 6.20392 14.5934 6.27565 14.6853C7.38876 16.1101 9.15759 17.75 12.0002 17.75C14.7604 17.75 16.5386 16.4804 17.7063 14.9195C17.7936 14.8027 17.9223 14.7449 18.0408 14.7427C18.1519 14.7406 18.2374 14.7844 18.2944 14.8705C19.0206 15.9692 19.9762 17.066 21.8282 17.5024C22.2313 17.5974 22.6352 17.3476 22.7302 16.9444C22.8252 16.5412 22.5754 16.1374 22.1722 16.0424Z"
                      fill="#f4a61f"
                    ></path>
                    <g opacity="0.5">
                      <path
                        d="M4.39936 4.39838C4.69225 4.10549 5.16712 4.10549 5.46002 4.39838L5.85285 4.79122C6.14575 5.08411 6.14575 5.55898 5.85285 5.85188C5.55996 6.14477 5.08509 6.14477 4.79219 5.85188L4.39936 5.45904C4.10646 5.16615 4.10646 4.69127 4.39936 4.39838Z"
                        fill="#f4a61f"
                      ></path>
                      <path
                        d="M19.6009 4.39864C19.8938 4.69153 19.8938 5.16641 19.6009 5.4593L19.2081 5.85214C18.9152 6.14503 18.4403 6.14503 18.1474 5.85214C17.8545 5.55924 17.8545 5.08437 18.1474 4.79148L18.5402 4.39864C18.8331 4.10575 19.308 4.10575 19.6009 4.39864Z"
                        fill="#f4a61f"
                      ></path>
                    </g>
                    <g opacity="0.5">
                      <path
                        d="M4.57004 18.8659C5.25688 17.7921 6.72749 17.8273 7.4577 18.762C8.4477 20.0293 9.82965 21.2502 12.0002 21.2502C14.2088 21.2502 15.5699 20.2714 16.5051 19.0211C17.2251 18.0587 18.7909 17.9015 19.5458 19.0437C20.1708 19.9893 20.8488 20.7308 22.1722 21.0426C22.5754 21.1376 22.8252 21.5414 22.7302 21.9446C22.6352 22.3478 22.2313 22.5976 21.8282 22.5026C19.9762 22.0662 19.0206 20.9694 18.2944 19.8707C18.2374 19.7846 18.1519 19.7408 18.0408 19.7429C17.9223 19.7451 17.7936 19.8029 17.7063 19.9197C16.5386 21.4806 14.7604 22.7502 12.0002 22.7502C9.15759 22.7502 7.38876 21.1103 6.27565 19.6855C6.20392 19.5936 6.11032 19.5587 6.02981 19.5604C5.95388 19.5621 5.88442 19.5948 5.83365 19.6742C5.09416 20.8303 4.13929 22.0391 2.17218 22.5026C1.76901 22.5976 1.36516 22.3478 1.27016 21.9446C1.17516 21.5414 1.42498 21.1376 1.82816 21.0426C3.2291 20.7125 3.90775 19.9013 4.57004 18.8659Z"
                        fill="#f4a61f"
                      ></path>
                    </g>
                  </g>
                </svg>
              </div>
              <h3>Plan Vacation</h3>
              <div className="example-card">
                <button
                  onClick={() =>
                    onExampleClick(
                      "Hi there, I want to have a relaxing summer vacation here. Can you make a list of recommended places where I should go?"
                    )
                  }
                >
                  "Hi there, I want to have a relaxing summer vacation here. Can
                  you make a list of recommended places where I should go?"
                </button>
              </div>
            </div>

            <div className="category">
              <div className="category-icon cloud">
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
                      fillRule="evenodd"
                      clipRule="evenodd"
                      d="M10.8539 14.5105C11.1243 14.8243 11.0891 15.2978 10.7753 15.5682L8.0195 17.9426H10.2857C10.5993 17.9426 10.8797 18.1376 10.9888 18.4315C11.098 18.7254 11.0128 19.0562 10.7753 19.2608L6.48957 22.9531C6.17575 23.2235 5.70219 23.1883 5.43183 22.8745C5.16147 22.5607 5.19669 22.0871 5.5105 21.8167L8.26616 19.4426H6.00004C5.68653 19.4426 5.4061 19.2476 5.29696 18.9537C5.18781 18.6598 5.27297 18.3291 5.51048 18.1244L9.79619 14.4318C10.11 14.1615 10.5836 14.1967 10.8539 14.5105ZM15.5304 14.9697C15.8233 15.2626 15.8233 15.7374 15.5304 16.0303L13.5304 18.0303C13.2375 18.3232 12.7626 18.3232 12.4697 18.0303C12.1768 17.7374 12.1768 17.2626 12.4697 16.9697L14.4697 14.9697C14.7626 14.6768 15.2375 14.6768 15.5304 14.9697ZM17.5304 18.4697C17.8233 18.7626 17.8233 19.2375 17.5304 19.5303L15.5304 21.5303C15.2375 21.8232 14.7626 21.8232 14.4697 21.5303C14.1768 21.2375 14.1768 20.7626 14.4697 20.4697L16.4697 18.4697C16.7626 18.1768 17.2375 18.1768 17.5304 18.4697ZM13.5304 19.4697C13.8233 19.7626 13.8233 20.2375 13.5304 20.5303L11.5304 22.5303C11.2375 22.8232 10.7626 22.8232 10.4697 22.5303C10.1768 22.2375 10.1768 21.7626 10.4697 21.4697L12.4697 19.4697C12.7626 19.1768 13.2375 19.1768 13.5304 19.4697Z"
                      fill="#2e62ff"
                    ></path>
                    <path
                      opacity="0.5"
                      d="M16.2857 19C19.4416 19 22 16.4717 22 13.3529C22 10.8811 20.393 8.78024 18.1551 8.01498C17.8371 5.19371 15.4159 3 12.4762 3C9.32028 3 6.7619 5.52827 6.7619 8.64706C6.7619 9.33687 6.88706 9.9978 7.11616 10.6089C6.8475 10.5567 6.56983 10.5294 6.28571 10.5294C3.91878 10.5294 2 12.4256 2 14.7647C2 17.1038 3.91878 19 6.28571 19H16.2857Z"
                      fill="#2e62ff"
                    ></path>
                  </g>
                </svg>
              </div>
              <h3>Brainstorming</h3>
              <div className="example-card">
                <button
                  onClick={() =>
                    onExampleClick(
                      "Can you give me some ideas and reasons on how to start a relationship between people?"
                    )
                  }
                >
                  "Can you give me some ideas and reasons on how to start a
                  relationship between people?"
                </button>
              </div>
            </div>

            <div className="category">
              <div className="category-icon document">
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
                      opacity="0.5"
                      d="M3 10C3 6.22876 3 4.34315 4.17157 3.17157C5.34315 2 7.22876 2 11 2H13C16.7712 2 18.6569 2 19.8284 3.17157C21 4.34315 21 6.22876 21 10V14C21 17.7712 21 19.6569 19.8284 20.8284C18.6569 22 16.7712 22 13 22H11C7.22876 22 5.34315 22 4.17157 20.8284C3 19.6569 3 17.7712 3 14V10Z"
                      fill="#616161"
                    ></path>
                    <path
                      fillRule="evenodd"
                      clipRule="evenodd"
                      d="M7.25 10C7.25 9.58579 7.58579 9.25 8 9.25H16C16.4142 9.25 16.75 9.58579 16.75 10C16.75 10.4142 16.4142 10.75 16 10.75H8C7.58579 10.75 7.25 10.4142 7.25 10Z"
                      fill="#616161"
                    ></path>
                    <path
                      fillRule="evenodd"
                      clipRule="evenodd"
                      d="M7.25 14C7.25 13.5858 7.58579 13.25 8 13.25H13C13.4142 13.25 13.75 13.5858 13.75 14C13.75 14.4142 13.4142 14.75 13 14.75H8C7.58579 14.75 7.25 14.4142 7.25 14Z"
                      fill="#616161"
                    ></path>
                  </g>
                </svg>
              </div>
              <h3>Case Analysis</h3>
              <div className="example-card">
                <button
                  onClick={() =>
                    onExampleClick(
                      "Why do I keep failing and being rejected compared to my friend? Does it mean that I will never catch up with the minimum requirements of the skillset of today, even if I did tried as hard as I can?"
                    )
                  }
                >
                  "Why do I keep failing and being rejected compared to my
                  friend? Does it mean that I will never catch up with the
                  minimum requirements of the skillset of today, even if I did
                  tried as hard as I can?"
                </button>
              </div>
            </div>
          </div>

          {/* <div className="input-placeholder">Enter a message here...</div> */}
        </div>
      ) : (
        <ul id="messages" className={fadeIn ? "fade-in" : ""}>
          {messages.map((message) => (
            <li
              key={message.id}
              className={`message-bubble ${message.sender === "user" ? "user-bubble" : "assistant-bubble"}`}
            >
              <div className="message-container">
                {/* Avatar Placeholder */}
                <div className={`avatar ${message.sender === "user" ? "user-avatar" : "assistant-avatar"}`}>
                  {message.sender === 'user' ? 'U' : 'A'} {/* Simple text avatar */}
                </div>
                <div className="message-content">
                  {renderMessageContent(message.content)}
                   {/* Interaction Buttons Placeholder (only for assistant) */}
                   {message.sender === 'assistant' && (
                    <div className="message-interactions">
                      {/* Buttons like Copy, Regenerate, Thumbs up/down will go here */}
                      <button title="Copy"><svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"></path></svg></button>
                      <button title="Regenerate"><svg viewBox="0 0 24 24"><path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"></path></svg></button>
                      {/* Add more button placeholders */}
                    </div>
                  )}
                </div>
              </div>
            </li>
          ))}
          {isGenerating && (
            <li className="message-bubble assistant-bubble">
               <div className="message-container">
                 <div className="avatar assistant-avatar">A</div>
                 <div className="message-content">
                    <div className="generating-indicator">
                      <span>Generating response</span>
                      <span className="dot-animation">...</span>
                    </div>
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

export default ChatFeed;
