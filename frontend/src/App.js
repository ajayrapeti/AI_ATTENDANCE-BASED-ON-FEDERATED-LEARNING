import './App.css';
import './index.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Header from './components/header/Header';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import TeacherDashboard from './pages/TeacherDashboard';
import { SessionProvider } from './contexts/SessionContext';
import ProtectedRoute from './components/ProtectedRoute';
import Proctordashboard from './pages/Proctordashboard';

function App() {
  return (
    <SessionProvider>
      <BrowserRouter>
        <Header />
        <Routes>
          {/* Auth routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          
          {/* Protected routes - more specific first */}
          <Route 
            path="/teacher-dashboard/*" 
            element={
              <ProtectedRoute allowedRoles={['teacher']}>
                <TeacherDashboard />
              </ProtectedRoute>
            } 
          />
          
          <Route 
            path="/proctor-dashboard/*" 
            element={
              <ProtectedRoute allowedRoles={['proctor']}>
                <Proctordashboard />
              </ProtectedRoute>
            } 
          />
          
          <Route 
            path="/dashboard/*" 
            element={
              <ProtectedRoute allowedRoles={['student']}>
                <Dashboard />
              </ProtectedRoute>
            } 
          />

          {/* Home route should be last */}
          <Route path="/" element={<Home />} />
        </Routes>
      </BrowserRouter>
    </SessionProvider>
  );
}

export default App;
