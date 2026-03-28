// src/hooks/useAuth.jsx
import { useState, useEffect, createContext, useContext } from 'react';
import { AuthAPI, TokenManager } from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadProfile();
    
    // Listen for auth expiration
    const handleExpired = () => {
      setUser(null);
    };
    window.addEventListener('auth-expired', handleExpired);
    return () => window.removeEventListener('auth-expired', handleExpired);
  }, []);

  async function loadProfile() {
    try {
      if (TokenManager.isValid()) {
        const profile = await AuthAPI.getProfile();
        setUser(profile);
      }
    } catch (err) {
      console.error('Profile load error:', err);
    } finally {
      setLoading(false);
    }
  }

  async function login(email, password) {
    await AuthAPI.login(email, password);
    const profile = await AuthAPI.getProfile();
    setUser(profile);
    return profile;
  }

  async function register(email, password, displayName) {
    await AuthAPI.register(email, password, displayName);
    const profile = await AuthAPI.getProfile();
    setUser(profile);
    return profile;
  }

  async function updateDisplayName(displayName) {
    const profile = await AuthAPI.updateProfile(displayName);
    setUser(profile);
    return profile;
  }

  function logout() {
    AuthAPI.logout();
    setUser(null);
  }

  return (
    <AuthContext.Provider value={{
      user,
      loading,
      isAuthenticated: !!user && !user.is_guest,
      isGuest: !user || user.is_guest,
      login,
      register,
      logout,
      updateDisplayName,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
