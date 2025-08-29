import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useSession } from "../../contexts/SessionContext";
import { motion, AnimatePresence } from "framer-motion";

// Finalized CircularDots Button Component
const CircularDotsButton = ({ isOpen, toggleMenu }) => (
  <motion.div 
    className="relative w-12 h-12 cursor-pointer"
    whileHover={{ scale: 1.1 }}
    whileTap={{ scale: 0.9 }}
  >
    {/* Expanding background circle */}
    <motion.div
      className="absolute inset-0 bg-primary-500 rounded-full"
      animate={{
        scale: isOpen ? 50 : 1,
        opacity: isOpen ? 0.2 : 1
      }}
      transition={{ duration: 0.5 }}
    />
    
    {/* Dots container */}
    <motion.div
      className="absolute inset-0 grid grid-cols-3 gap-1 p-3"
      onClick={toggleMenu}
    >
      {[...Array(9)].map((_, i) => (
        <motion.div
          key={i}
          className="w-2 h-2 bg-white rounded-full"
          animate={{
            scale: isOpen ? [1, 0.5, 1] : 1,
            opacity: isOpen ? [1, 0.5, 1] : 1
          }}
          transition={{
            duration: 0.5,
            delay: i * 0.05,
            repeat: isOpen ? Infinity : 0
          }}
        />
      ))}
    </motion.div>
  </motion.div>
);

export default function Header() {
  const { session, clearSession } = useSession();
  const navigate = useNavigate();
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleLogout = () => {
    clearSession();
    navigate('/login');
  };

  return (
    <nav 
      className={`fixed w-full top-0 left-0 right-0 z-50 transition-all duration-300 m-0 ${
        isScrolled ? 'py-0' : 'py-0'
      }`}
    >
      <div className={`
        backdrop-blur-md 
        ${isScrolled ? 'bg-secondary-900/90' : 'bg-transparent'} 
        transition-all duration-300
        m-0
      `}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo/Brand with animation */}
            <motion.div 
              className="flex-shrink-0"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Link to="/" className="text-xl font-semibold text-primary-100">
                AttendanceApp
              </Link>
            </motion.div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex space-x-8">
              <NavLinks session={session} handleLogout={handleLogout} />
            </div>

            {/* Mobile Navigation Button */}
            <div className="md:hidden">
              <CircularDotsButton 
                isOpen={isMobileMenuOpen} 
                toggleMenu={() => setIsMobileMenuOpen(!isMobileMenuOpen)} 
              />
            </div>
          </div>
        </div>

        {/* Mobile Menu */}
        <AnimatePresence>
          {isMobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden"
            >
              <div className="px-2 pt-2 pb-3 space-y-1 backdrop-blur-lg bg-secondary-900/90">
                <MobileNavLinks session={session} handleLogout={handleLogout} setIsMobileMenuOpen={setIsMobileMenuOpen} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </nav>
  );
}

// NavLinks component
const NavLinks = ({ session, handleLogout }) => (
  <>
    <Link 
      to="/" 
      className="text-secondary-300 hover:text-primary-300 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
    >
      Home
    </Link>
    {session ? (
      <>
        <Link 
          to="/dashboard" 
          className="text-secondary-300 hover:text-primary-300 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
        >
          Dashboard
        </Link>
        <motion.button 
          onClick={handleLogout}
          className="text-secondary-300 hover:text-primary-300 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          Logout
        </motion.button>
      </>
    ) : (
      <Link 
        to="/login" 
        className="text-secondary-300 hover:text-primary-300 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
      >
        Login
      </Link>
    )}
  </>
);

// MobileNavLinks component
const MobileNavLinks = ({ session, handleLogout, setIsMobileMenuOpen }) => {
  const closeMenu = () => setIsMobileMenuOpen(false);
  
  return (
    <>
      <motion.div
        whileTap={{ scale: 0.95 }}
        className="block w-full"
      >
        <Link 
          to="/" 
          onClick={closeMenu}
          className="block px-3 py-2 rounded-md text-base font-medium text-secondary-300 hover:text-primary-300 hover:bg-secondary-800 transition-colors duration-200"
        >
          Home
        </Link>
      </motion.div>
      {session ? (
        <>
          <motion.div
            whileTap={{ scale: 0.95 }}
            className="block w-full"
          >
            <Link 
              to="/dashboard" 
              onClick={closeMenu}
              className="block px-3 py-2 rounded-md text-base font-medium text-secondary-300 hover:text-primary-300 hover:bg-secondary-800 transition-colors duration-200"
            >
              Dashboard
            </Link>
          </motion.div>
          <motion.div
            whileTap={{ scale: 0.95 }}
            className="block w-full"
          >
            <button
              onClick={() => {
                handleLogout();
                closeMenu();
              }}
              className="w-full text-left px-3 py-2 rounded-md text-base font-medium text-secondary-300 hover:text-primary-300 hover:bg-secondary-800 transition-colors duration-200"
            >
              Logout
            </button>
          </motion.div>
        </>
      ) : (
        <motion.div
          whileTap={{ scale: 0.95 }}
          className="block w-full"
        >
          <Link 
            to="/login" 
            onClick={closeMenu}
            className="block px-3 py-2 rounded-md text-base font-medium text-secondary-300 hover:text-primary-300 hover:bg-secondary-800 transition-colors duration-200"
          >
            Login
          </Link>
        </motion.div>
      )}
    </>
  );
};
