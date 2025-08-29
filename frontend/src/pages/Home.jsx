import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useSession } from '../contexts/SessionContext';

const Home = () => {
  const { session, loading } = useSession();
  const navigate = useNavigate();

  const getDashboardPath = () => {
    if (!session?.user) return '/login';   
    console.log("session.user.role",session.user.role);
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

  const handleDashboardClick = () => {
    if (loading) return;
    const path = getDashboardPath();
    navigate(path);
  };
  

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative h-screen flex items-center justify-center bg-[url('https://images.unsplash.com/photo-1497864149936-d3163f0c0f4b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80')] bg-cover bg-center">
        {/* Dark overlay */}
        <div className="absolute inset-0 bg-black bg-opacity-50"></div>
        
        {/* Content */}
        <div className="container mx-auto px-6 text-center relative z-10">
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-6 animate-fade-in">
            Smart Attendance Management
          </h1>
          <p className="text-xl md:text-2xl text-gray-200 mb-12 max-w-3xl mx-auto">
            Streamline your attendance tracking with our modern, intelligent solution
          </p>
          <button 
            onClick={handleDashboardClick}
            disabled={loading}
            className={`inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg transition duration-300 transform hover:scale-105 ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {loading ? 'Loading...' : session?.user ? 'Go to Dashboard' : 'Get Started'}
          </button>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            <div className="p-6 bg-white rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-300">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-6">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-4">Real-time Tracking</h3>
              <p className="text-gray-600">Monitor attendance in real-time with our advanced tracking system</p>
            </div>
            
            <div className="p-6 bg-white rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-300">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-6">
                <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-4">Automated Reports</h3>
              <p className="text-gray-600">Generate detailed reports with just a few clicks</p>
            </div>

            <div className="p-6 bg-white rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-300">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mb-6">
                <svg className="w-8 h-8 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-4">Secure Access</h3>
              <p className="text-gray-600">Enterprise-grade security for your attendance data</p>
            </div>
          </div>
        </div>
      </section>

      {/* Coming Soon Section */}
      <section className="py-20 bg-gray-900 text-white">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold mb-8">Coming Soon</h2>
          <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="p-6 bg-gray-800 rounded-lg">
              <h3 className="text-xl font-semibold mb-4">Mobile App</h3>
              <p className="text-gray-400">Track attendance on the go with our upcoming mobile application</p>
            </div>
            <div className="p-6 bg-gray-800 rounded-lg">
              <h3 className="text-xl font-semibold mb-4">AI Analytics</h3>
              <p className="text-gray-400">Advanced insights powered by artificial intelligence</p>
            </div>
          </div>
          <div className="mt-12">
            <input
              type="email"
              placeholder="Enter your email for updates"
              className="px-6 py-3 rounded-lg mr-4 bg-gray-800 text-white border border-gray-700 focus:outline-none focus:border-blue-500"
            />
            <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg transition duration-300">
              Notify Me
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;