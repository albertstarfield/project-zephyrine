import React, { createContext, useState, useEffect, useContext } from 'react';
// Removed: import { supabase } from '../utils/supabaseClient'; // No longer using Supabase client here

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  // No need for loading state as we are providing a static dummy user
  // const [loading, setLoading] = useState(true); // Removed

  // Define a static dummy user and session
  // You can customize the ID/email if needed elsewhere in your app
  const dummyUser = {
    id: 'local-user-123', // A unique ID for the local user
    email: 'local@example.com',
    // Add other user properties if your components rely on them
    // e.g., app_metadata: { provider: 'email' }, user_metadata: { name: 'Local User' }
  };

  const dummySession = {
    user: dummyUser,
    access_token: 'dummy-local-token-abc-123', // Placeholder token if needed anywhere
    token_type: 'bearer',
    expires_in: 3600, // Placeholder
    expires_at: Math.floor(Date.now() / 1000) + 3600, // Placeholder
    // Add other session properties if needed
  };

  // The useEffect for checking/listening to Supabase auth is removed
  // useEffect(() => { ... supabase calls ... }, []); // Removed

  // Define the context value with dummy data and functions
  const value = {
    session: dummySession, // Provide the static dummy session
    user: dummyUser,       // Provide the static dummy user
    signIn: async (email, password) => {
      console.log("AuthContext: Dummy signIn called for", email);
      // Simulate successful sign-in immediately by returning the dummy session/user
      // In a real non-supabase scenario, this would call your backend
      return { data: { session: dummySession, user: dummyUser }, error: null };
    },
    signUp: async (email, password) => {
      console.log("AuthContext: Dummy signUp called for", email);
      // Simulate successful sign-up
      // In a real non-supabase scenario, this would call your backend
      return { data: { session: dummySession, user: dummyUser }, error: null };
    },
    signOut: () => {
      console.log("AuthContext: Dummy signOut called.");
      // In this dummy setup, signOut does nothing to the context state
      // A more complex dummy might set session/user to null
    },
    loading: false, // Always false, initial check is skipped
  };

  return (
    <AuthContext.Provider value={value}>
      {/* Render children immediately since loading is always false */}
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use the AuthContext remains the same
export const useAuth = () => {
  return useContext(AuthContext);
};