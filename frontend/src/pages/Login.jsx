import React, { useState } from 'react';
import { Link, Navigate } from 'react-router-dom';
import axios from 'axios';
import { useSession } from '../contexts/SessionContext';

const Login = () => {
  const { session, updateSession, loading } = useSession();
  const [formData, setFormData] = useState({
    rollNumber: '',
    password: ''
  });
  const [isLoading, setIsLoading] = useState(false);

  // If loading, show loading spinner
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  // If there's an active session, redirect based on role
  if (session?.user) {
    if (session.user.role === 'teacher') {
      return <Navigate to="/teacher-dashboard" replace />;
    } else if (session.user.role === 'proctor') {
      return <Navigate to="/proctor-dashboard" replace />;
    } else if (session.user.role === 'student') {
      return <Navigate to="/dashboard" replace />;
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      const response = await axios.post('http://localhost:3000/auth/login', {
        rollNumber: formData.rollNumber,
        password: formData.password
      });

      const { access_token, user } = response.data.data;
      updateSession(access_token, user);
    } catch (error) {
      console.error('Login failed:', error);
      const errorMessage = error.response?.data?.message || 'Login failed';
      alert(Array.isArray(errorMessage) ? errorMessage.join('\n') : errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-secondary-900 to-primary-900">
      <div className="max-w-md w-full space-y-8 p-10 bg-secondary-800/50 backdrop-blur-md rounded-xl border border-secondary-700/30 shadow-2xl">
        <div>
          <h2 className="text-3xl font-bold text-center text-primary-100">
           Login
          </h2>
          {/* <p className="mt-2 text-center text-secondary-300">
            Or{' '}
            <Link to="/register" className="text-primary-400 hover:text-primary-300">
              register as a new student
            </Link>
          </p> */}
        </div>
        
        <form onSubmit={handleSubmit} className="mt-8 space-y-6">
          <input type="hidden" name="remember" value="true" />
          <div className="space-y-4">
            <div>
              <label htmlFor="roll-number" className="text-secondary-200">
                Roll Number
              </label>
              <input
                id="roll-number"
                name="rollNumber"
                type="text"
                required
                className="appearance-none rounded-lg relative block w-full px-3 py-2 border border-secondary-700 bg-secondary-900/50 placeholder-secondary-400 text-secondary-100 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="Enter your roll number"
                value={formData.rollNumber}
                onChange={(e) => setFormData({ ...formData, rollNumber: e.target.value })}
              />
            </div>
            <div>
              <label htmlFor="password" className="text-secondary-200">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                required
                className="appearance-none rounded-lg relative block w-full px-3 py-2 border border-secondary-700 bg-secondary-900/50 placeholder-secondary-400 text-secondary-100 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="Enter your password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <input
                id="remember-me"
                name="remember-me"
                type="checkbox"
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-600 rounded bg-secondary-900/50"
              />
              <label htmlFor="remember-me" className="ml-2 block text-sm text-secondary-200">
                Remember me
              </label>
            </div>

            <div className="text-sm">
              <Link to="/forgot-password" className="font-medium text-primary-400 hover:text-primary-300 transition-colors">
                Forgot password?
              </Link>
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-lg text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? "Signing in..." : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default Login;