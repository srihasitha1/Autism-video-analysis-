// src/components/BottomTabBar.jsx
import React from 'react';
import { Home, BarChart2, MapPin, Clock, User } from 'lucide-react';

export default function BottomTabBar({ current, onNavigate, onLogout }) {
  const tabs = [
    { id: 'assessment', icon: Home, label: 'Home' },
    { id: 'results', icon: BarChart2, label: 'Results' },
    { id: 'clinics', icon: MapPin, label: 'Clinics' },
    { id: 'history', icon: Clock, label: 'History' },
    { id: 'profile', icon: User, label: 'Profile' },
  ];

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-100 shadow-[0_-4px_20px_rgba(0,0,0,0.05)] z-50 md:relative md:w-64 md:h-screen md:border-r md:border-t-0 md:shadow-lg">
      <nav className="flex justify-around items-center h-16 md:flex-col md:h-full md:py-6 md:px-3 md:justify-start md:gap-2">
        {/* Logo for desktop */}
        <div className="hidden md:flex items-center gap-2 mb-6 px-2">
          <div className="w-10 h-10 bg-teal-100 rounded-xl flex items-center justify-center">
            <span className="text-[#1A9E7A] font-bold text-lg">A</span>
          </div>
          <span className="font-bold text-gray-800 text-lg">AutiSense</span>
        </div>

        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = current === tab.id;
          
          return (
            <button
              key={tab.id}
              onClick={() => {
                if (tab.id === 'assessment') {
                  onNavigate('home');
                } else {
                  onNavigate(tab.id);
                }
              }}
              className={`flex flex-col items-center justify-center py-2 px-3 rounded-xl transition-all ${
                isActive 
                  ? 'text-[#1A9E7A] bg-teal-50 md:bg-teal-100/50' 
                  : 'text-gray-400 hover:text-gray-600 hover:bg-gray-50'
              } md:w-full md:flex-row md:gap-3 md:py-3`}
            >
              <Icon className={`w-5 h-5 ${isActive ? 'text-[#1A9E7A]' : ''}`} />
              <span className={`text-xs font-medium mt-0.5 md:text-sm md:mt-0 ${isActive ? 'font-bold' : ''}`}>
                {tab.label}
              </span>
            </button>
          );
        })}

        {/* Spacer for desktop */}
        <div className="hidden md:block flex-1"></div>

        {/* Logout for desktop */}
        <button
          onClick={onLogout}
          className="hidden md:flex items-center gap-2 px-4 py-2 text-gray-500 hover:text-red-500 hover:bg-red-50 rounded-xl transition-all w-full"
        >
          <span className="text-sm font-medium">Log Out</span>
        </button>
      </nav>
    </div>
  );
}
