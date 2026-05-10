// ExternalAnalyzer/frontend-face-zephyrine/src/components/ConversationTurn.jsx
import React from 'react';
import PropTypes from 'prop-types';

const ConversationTurn = ({ userText, assistantText }) => {
  return (
    <div className="voice-conversation-turn">
      {userText && (
        <div className="voice-turn-bubble user">
          <span className="voice-turn-speaker">You</span>
          <p>{userText}</p>
        </div>
      )}
      {assistantText && (
        <div className="voice-turn-bubble assistant">
          <span className="voice-turn-speaker">Zephyrine</span>
          <p>{assistantText}</p>
        </div>
      )}
    </div>
  );
};

ConversationTurn.propTypes = {
  userText: PropTypes.string,
  assistantText: PropTypes.string,
};

export default ConversationTurn;