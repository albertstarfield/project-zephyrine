import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
// Consider creating a separate CSS file for Auth component styles
// import '../styles/Auth.css';

const Auth = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const { signIn, signUp } = useAuth();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    const { error } = await signIn(email, password);
    if (error) {
      setError(error.message);
    }
    // No need to redirect here, AuthContext listener will update the state in App.jsx
    setLoading(false);
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    const { error } = await signUp(email, password);
    if (error) {
      setError(error.message);
    } else {
      // Optionally, display a message asking the user to check their email for verification
      alert('Signup successful! Please check your email to verify your account.');
    }
    setLoading(false);
  };

  return (
    <div className="auth-container" style={styles.container}> {/* Added inline styles for basic layout */}
      <h1 style={styles.header}>Welcome to Zephyrine</h1>
      <p style={styles.subHeader}>Please sign in or sign up to continue</p>
      <form style={styles.form}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          style={styles.input}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          style={styles.input}
        />
        {error && <p style={styles.error}>{error}</p>}
        <div style={styles.buttonGroup}>
          <button type="submit" onClick={handleLogin} disabled={loading} style={styles.button}>
            {loading ? 'Signing In...' : 'Sign In'}
          </button>
          <button type="button" onClick={handleSignup} disabled={loading} style={{...styles.button, ...styles.buttonSecondary}}>
            {loading ? 'Signing Up...' : 'Sign Up'}
          </button>
        </div>
      </form>
    </div>
  );
};

// Basic inline styles - consider moving to a CSS file
const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '100vh',
    // Use background variables if defined in your CSS
    // background: 'var(--background-color, #f0f0f0)',
    padding: '20px',
  },
  header: {
    marginBottom: '10px',
    // color: 'var(--text-color, #333)',
  },
  subHeader: {
    marginBottom: '30px',
    // color: 'var(--text-secondary-color, #666)',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    width: '100%',
    maxWidth: '300px',
    background: '#fff', // Example background
    padding: '20px',
    borderRadius: '8px',
    boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
  },
  input: {
    marginBottom: '15px',
    padding: '10px',
    border: '1px solid #ccc',
    borderRadius: '4px',
    fontSize: '1rem',
  },
  buttonGroup: {
    display: 'flex',
    justifyContent: 'space-between',
    marginTop: '10px',
  },
  button: {
    padding: '10px 15px',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '1rem',
    background: '#007bff', // Example primary color
    color: '#fff',
    flex: 1, // Make buttons take equal space
    margin: '0 5px', // Add some space between buttons
  },
  buttonSecondary: {
     background: '#6c757d', // Example secondary color
  },
  error: {
    color: 'red',
    marginBottom: '10px',
    textAlign: 'center',
  },
};


export default Auth;
