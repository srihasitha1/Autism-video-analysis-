import React from 'react';
import { AuthProvider } from './hooks/useAuth';
import AppContent from './AppContent';

export default function AutiSenseApp() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
