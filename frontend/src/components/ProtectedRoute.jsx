import { Navigate } from 'react-router-dom';
import { useSession } from '../contexts/SessionContext';
import { motion } from 'framer-motion';

export default function ProtectedRoute({ children, allowedRoles = [] }) {
  const { session, loading } = useSession();

  // Debug logs
  console.log('Protected Route Session:', session);
  console.log('Allowed Roles:', allowedRoles);
  console.log('Current User Role:', session?.user?.role);

  const getDashboardPath = () => {
    if (!session?.user) return '/login';
    console.log("session.user.role", session.user.role);
    switch (session.user.role) {
      case 'teacher':
        return '/teacher-dashboard';
      case 'proctor':
        return '/proctor-dashboard';
      case 'student':
        return '/dashboard';
      default:
        return '/login';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-secondary-900 to-primary-900 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{
            duration: 0.8,
            ease: [0, 0.71, 0.2, 1.01],
            scale: {
              type: "spring",
              damping: 5,
              stiffness: 100,
              restDelta: 0.001
            }
          }}
        >
          <div className="bg-secondary-800/50 backdrop-blur-md rounded-xl border border-secondary-700/30 p-8 shadow-lg">
            <div className="flex flex-col items-center gap-4">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-400"></div>
              <p className="text-primary-100 text-sm">Loading...</p>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  // Check if user is authenticated
  if (!session || !session.token) {
    console.log('No session or token, redirecting to login');
    return <Navigate to="/login" replace />;
  }

  // Check if user has required role
  if (allowedRoles.length > 0 && !allowedRoles.includes(session.user.role)) {
    console.log('User role not allowed, redirecting to appropriate dashboard');
    return <Navigate to={getDashboardPath()} replace />;
  }

  // If all checks pass, render the protected component
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      {children}
    </motion.div>
  );
} 