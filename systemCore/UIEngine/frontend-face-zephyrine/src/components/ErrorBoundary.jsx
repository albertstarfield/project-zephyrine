import React from 'react';
import PropTypes from 'prop-types';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI.
    return { hasError: true, error: error };
  }

  componentDidCatch(error, errorInfo) {
    // You can also log the error to an error reporting service
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      // You can render any custom fallback UI
      return (
        <div style={{
          padding: '1rem',
          backgroundColor: 'rgba(255, 93, 93, 0.1)',
          border: '1px solid var(--error)',
          borderRadius: 'var(--rounded-border)',
          color: 'var(--primary-text)',
        }}>
          <p style={{ fontWeight: 'bold' }}>Something went wrong rendering this part of the message.</p>
          {/* Optionally, you can display the error message during development */}
          <pre style={{
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all',
            fontSize: '0.8rem',
            color: 'var(--secondary-text)',
            marginTop: '1rem',
          }}>
            {this.state.error.toString()}
          </pre>
        </div>
      );
    }

    return this.props.children;
  }
}

ErrorBoundary.propTypes = {
  children: PropTypes.node.isRequired,
};

export default ErrorBoundary;
