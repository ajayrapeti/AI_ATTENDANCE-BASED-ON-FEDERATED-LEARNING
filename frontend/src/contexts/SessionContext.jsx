import React, { createContext, useContext, useState, useEffect } from 'react';

const SessionContext = createContext();

export function SessionProvider({ children }) {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check for existing session on mount
    const token = sessionStorage.getItem('token');
    const user = sessionStorage.getItem('user');
    if (token && user) {
      setSession({ token, user: JSON.parse(user) });
    }
    setLoading(false);
  }, []);

  const updateSession = (token, user) => {
    console.log(token, user);
    sessionStorage.setItem('token', token);
    sessionStorage.setItem('user', JSON.stringify(user));
    setSession({ token, user });
  };

  const clearSession = () => {
    sessionStorage.removeItem('token');
    sessionStorage.removeItem('user');
    setSession(null);
  };

  return (
    <SessionContext.Provider value={{ session, loading, updateSession, clearSession }}>
      {children}
    </SessionContext.Provider>
  );
}

export const useSession = () => useContext(SessionContext); 