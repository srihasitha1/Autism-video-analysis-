import React, { useState, useEffect, useRef } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import {
  Activity, Brain, Heart, Search, MessageSquare, Video,
  ClipboardList, BarChart2, MapPin, ChevronRight, ArrowLeft,
  Send, User, Home, Calendar, Settings, AlertTriangle, Check, Star
} from 'lucide-react';

const PrimaryColor = "text-[#1A9E7A]";
const PrimaryBg = "bg-[#1A9E7A]";
const AccentBg = "bg-[#2ECC71]";
const AccentText = "text-[#2ECC71]";

export default function AutiSenseApp() {
  const [currentScreen, setCurrentScreen] = useState('login');
  const [assessmentStatus, setAssessmentStatus] = useState({ video: false, questionnaire: false });
  const [sessionUUID, setSessionUUID] = useState(null);
  const [authToken, setAuthToken] = useState(null);
  
  // Navigation helper
  const navigateTo = (screen) => setCurrentScreen(screen);
  const markComplete = (task) => setAssessmentStatus(prev => ({ ...prev, [task]: true }));

  // Create a guest session and store the UUID
  const createGuestSession = async () => {
    try {
      const res = await fetch('/api/v1/auth/guest', { method: 'POST' });
      if (!res.ok) throw new Error('Failed to create session');
      const data = await res.json();
      setSessionUUID(data.session_uuid);
      return data.session_uuid;
    } catch (err) {
      console.error('Session creation failed:', err);
      return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex justify-center font-sans" style={{ fontFamily: "'DM Sans', sans-serif" }}>
      {/* Injecting Fonts */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Sora:wght@400;600;700&display=swap');
        h1, h2, h3, h4, h5, h6 { font-family: 'Sora', sans-serif; }
      `}</style>

      {/* Main App Container - Now fully responsive! */}
      <div className={`w-full max-w-full bg-white shadow-xl min-h-screen relative flex flex-col md:flex-row overflow-hidden`}>
        
        {/* Responsive Sidebar / Bottom Tab Bar */}
        {['home', 'assessment_hub', 'results', 'clinics', 'history', 'profile'].includes(currentScreen) && (
          <BottomTabBar current={currentScreen === 'home' || currentScreen === 'assessment_hub' ? 'assessment' : currentScreen} onNavigate={navigateTo} />
        )}
        
        {/* Content Area */}
        <div className={`flex-1 relative h-screen overflow-hidden flex flex-col ${['home', 'assessment_hub', 'results', 'clinics', 'history', 'profile'].includes(currentScreen) ? 'md:w-[calc(100%-16rem)]' : 'w-full'}`}>
          {currentScreen === 'login' && <LoginScreen onNavigate={navigateTo} createGuestSession={createGuestSession} setAuthToken={setAuthToken} />}
          {currentScreen === 'home' && <HomeScreen onNavigate={navigateTo} />}
          {currentScreen === 'assessment_hub' && <AssessmentHubScreen onNavigate={navigateTo} status={assessmentStatus} />}
          {currentScreen === 'video_analysis' && <VideoAnalysisScreen onNavigate={navigateTo} onComplete={() => markComplete('video')} sessionUUID={sessionUUID} />}
          {currentScreen === 'questionnaire' && <QuestionnaireScreen onNavigate={navigateTo} onComplete={() => markComplete('questionnaire')} sessionUUID={sessionUUID} />}
          {currentScreen === 'results' && <ResultsScreen onNavigate={navigateTo} sessionUUID={sessionUUID} />}
          {currentScreen === 'chatbot' && <ChatbotScreen onNavigate={navigateTo} />}
          {currentScreen === 'clinics' && <ClinicsScreen onNavigate={navigateTo} />}
          {currentScreen === 'create_account' && <CreateAccountScreen onNavigate={navigateTo} createGuestSession={createGuestSession} setAuthToken={setAuthToken} />}
          {currentScreen === 'history' && <HistoryScreen onNavigate={navigateTo} />}
          {currentScreen === 'profile' && <ProfileScreen onNavigate={navigateTo} />}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------
// 1. LOGIN / WELCOME SCREEN
// ---------------------------------------------------------
function LoginScreen({ onNavigate, createGuestSession, setAuthToken }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLogin = async () => {
    if (!email || !password) { setError('Please enter email and password'); return; }
    setLoading(true); setError('');
    try {
      const res = await fetch('/api/v1/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || 'Invalid credentials');
      }
      const data = await res.json();
      setAuthToken(data.access_token);
      // Also create an assessment session for this logged-in user
      await createGuestSession();
      onNavigate('home');
    } catch (err) {
      setError(err.message);
    } finally { setLoading(false); }
  };

  const handleGuest = async () => {
    setLoading(true); setError('');
    const uuid = await createGuestSession();
    setLoading(false);
    if (uuid) { onNavigate('home'); }
    else { setError('Could not connect to server. Is the backend running?'); }
  };

  return (
    <div className="flex-1 bg-gradient-to-br from-teal-50 to-white relative flex flex-col overflow-y-auto overflow-x-hidden w-full h-full">
      {/* Background shape */}
      <div className="fixed top-0 left-0 w-full h-1/2 md:w-1/2 md:h-full bg-teal-100/40 rounded-b-[100%] md:rounded-b-none md:rounded-r-[100%] scale-150 -translate-y-1/4 md:translate-y-0 md:scale-110 md:-translate-x-10 origin-top md:origin-left pointer-events-none z-0"></div>

      <div className="flex-1 flex flex-col items-center justify-center p-6 min-h-[600px] w-full max-w-7xl mx-auto relative z-10">
        
        <div className="w-full max-w-sm md:max-w-md bg-white/60 backdrop-blur-xl p-8 md:p-10 rounded-[2rem] shadow-xl border border-white/50 flex flex-col items-center text-center">
          <div className="w-20 h-20 bg-white shadow-md rounded-2xl flex items-center justify-center mb-6 relative border border-gray-50">
            <Brain className="w-10 h-10 text-[#1A9E7A]" />
            <Heart className="w-5 h-5 text-[#2ECC71] absolute right-2 bottom-2" fill="currentColor" />
          </div>

          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-2">AutiSense</h1>
          <p className="text-gray-500 font-medium mb-10 text-center">Early awareness. Better futures.</p>

          {error && (
            <div className="w-full bg-red-50 border border-red-200 text-red-600 text-sm px-4 py-3 rounded-xl mb-4 font-medium">
              {error}
            </div>
          )}

          <div className="w-full space-y-4 mb-8">
            <div className="relative">
              <User className="absolute left-4 top-3.5 w-5 h-5 text-gray-400" />
              <input
                type="email"
                placeholder="Email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-white/80 border border-gray-200 rounded-2xl py-3.5 pl-12 pr-4 text-gray-700 focus:outline-none focus:ring-2 focus:ring-[#1A9E7A] focus:bg-white transition-all shadow-sm"
              />
            </div>
            <div className="relative">
              <svg className="absolute left-4 top-3.5 w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8V7a4 4 0 00-8 0v4h8z" />
              </svg>
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-white/80 border border-gray-200 rounded-2xl py-3.5 pl-12 pr-4 text-gray-700 focus:outline-none focus:ring-2 focus:ring-[#1A9E7A] focus:bg-white transition-all shadow-sm"
              />
            </div>
            <div className="flex justify-end w-full">
              <button className="text-sm font-bold text-[#1A9E7A] hover:text-teal-800 transition-colors">Forgot Password?</button>
            </div>
          </div>

          <button
            onClick={handleLogin}
            disabled={loading}
            className="w-full bg-[#1A9E7A] hover:bg-teal-700 text-white font-bold py-4 rounded-full shadow-lg shadow-teal-200 hover:shadow-xl transition-all mb-4 text-lg disabled:opacity-50"
          >
            {loading ? 'Connecting...' : 'Sign In'}
          </button>

          <button 
            onClick={() => onNavigate('create_account')}
            className="w-full text-[#1A9E7A] font-bold py-3.5 rounded-full mb-2 hover:bg-teal-50 transition-colors border-2 border-transparent hover:border-teal-100"
          >
            Create Account
          </button>
          <button
            onClick={handleGuest}
            disabled={loading}
            className="w-full text-gray-500 font-bold py-3.5 rounded-full hover:bg-gray-50 transition-all text-sm border-2 border-transparent disabled:opacity-50"
          >
            {loading ? 'Creating session...' : 'Continue as Guest'}
          </button>
        </div>
        
        <div className="mt-8 text-center text-[11px] md:text-sm text-gray-400 px-4 max-w-sm font-medium">
          This app provides screening tools and risk assessment, not a formal medical diagnosis.
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------
// 2. HOME / DASHBOARD SCREEN
// ---------------------------------------------------------
function HomeScreen({ onNavigate }) {
  return (
    <div className="flex-1 bg-gray-50 pb-20 overflow-y-auto overflow-x-hidden flex flex-col">
      {/* Header */}
      <div className="bg-white px-6 py-5 border-b border-gray-100 shadow-sm z-10 relative">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-teal-100 rounded-full border-2 border-white shadow-sm flex items-center justify-center overflow-hidden">
              <span className="text-teal-700 font-bold text-lg">JS</span>
            </div>
            <div>
              <p className="text-sm text-gray-500 font-medium">Welcome back,</p>
              <h2 className="text-xl font-bold text-gray-900">Jane's Parent</h2>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 p-6 flex flex-col items-center justify-center">
        {/* Take Test Card Centered */}
        <div className="bg-gradient-to-br from-[#1A9E7A] to-teal-800 rounded-3xl p-8 md:p-12 text-white shadow-lg relative overflow-hidden w-full max-w-lg md:max-w-2xl transform transition-transform hover:scale-[1.02]">
          <div className="absolute -right-6 -top-6 w-32 h-32 bg-white/10 rounded-full blur-2xl"></div>
          <div className="absolute right-10 bottom-4 w-16 h-16 bg-[#2ECC71]/40 rounded-full blur-xl"></div>

          <div className="flex justify-between items-start mb-6 relative z-10">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#2ECC71] opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-[#2ECC71]"></span>
                </span>
                <p className="text-teal-100 text-sm font-medium">Ready</p>
              </div>
              <h3 className="text-3xl font-bold mb-2">Take Test</h3>
            </div>
            <div className="w-14 h-14 bg-white/20 rounded-2xl flex items-center justify-center backdrop-blur-sm shadow-sm border border-white/10">
              <ClipboardList className="w-7 h-7 text-white" />
            </div>
          </div>
          <p className="text-teal-50 text-[15px] mb-8 max-w-sm leading-relaxed relative z-10">
            Launch the guided setup to begin evaluating behavior patterns. Completing the video and questionnaire provides an accurate risk assessment.
          </p>
          <button
            onClick={() => onNavigate('assessment_hub')}
            className="bg-[#2ECC71] hover:bg-green-400 text-white font-bold py-4 px-6 rounded-full w-full shadow-lg transition-all relative z-10 text-lg flex items-center justify-center gap-2"
          >
            Start Test <ChevronRight className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------
// 2.5 ASSESSMENT HUB SCREEN
// ---------------------------------------------------------
function AssessmentHubScreen({ onNavigate, status }) {
  const isReady = status.video && status.questionnaire;

  return (
    <div className="flex-1 bg-gray-50 flex flex-col h-full overflow-hidden">
      <div className="px-6 py-5 bg-white border-b border-gray-100 shadow-sm flex items-center gap-4 z-10 relative">
        <button onClick={() => onNavigate('home')} className="w-10 h-10 rounded-full bg-gray-50 flex items-center justify-center text-gray-600 hover:bg-gray-100">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <h2 className="text-xl font-bold text-gray-900">Assessment Hub</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-6 md:p-10 md:max-w-4xl md:mx-auto w-full">
        <p className="text-gray-500 mb-8 font-medium">Please complete both modules below to generate your child's comprehensive risk analysis.</p>

        <div className="space-y-4 mb-10">
          {/* Card 1: Video */}
          <button 
            onClick={() => onNavigate('video_analysis')}
            className={`w-full bg-white p-5 rounded-3xl shadow-sm border-2 text-left flex items-center justify-between transition-all ${status.video ? 'border-green-400 bg-green-50/30' : 'border-transparent hover:border-teal-200 hover:shadow-md'}`}
          >
            <div className="flex items-center gap-4">
              <div className={`w-14 h-14 rounded-2xl flex items-center justify-center ${status.video ? 'bg-green-100 text-green-600' : 'bg-blue-100 text-blue-500'}`}>
                {status.video ? <Check className="w-7 h-7" /> : <Video className="w-7 h-7" />}
              </div>
              <div>
                <h3 className="font-bold text-gray-900 text-lg">1. Video Analysis</h3>
                <p className="text-sm text-gray-500">{status.video ? 'Completed successfully' : 'Record a 2-minute behavioral video'}</p>
              </div>
            </div>
            {!status.video && <ChevronRight className="w-6 h-6 text-gray-300" />}
          </button>

          {/* Card 2: Questionnaire */}
          <button 
            onClick={() => onNavigate('questionnaire')}
            className={`w-full bg-white p-5 rounded-3xl shadow-sm border-2 text-left flex items-center justify-between transition-all ${status.questionnaire ? 'border-green-400 bg-green-50/30' : 'border-transparent hover:border-teal-200 hover:shadow-md'}`}
          >
            <div className="flex items-center gap-4">
              <div className={`w-14 h-14 rounded-2xl flex items-center justify-center ${status.questionnaire ? 'bg-green-100 text-green-600' : 'bg-orange-100 text-orange-500'}`}>
                {status.questionnaire ? <Check className="w-7 h-7" /> : <ClipboardList className="w-7 h-7" />}
              </div>
              <div>
                <h3 className="font-bold text-gray-900 text-lg">2. Questionnaire</h3>
                <p className="text-sm text-gray-500">{status.questionnaire ? 'Completed successfully' : 'Answer 40 behavioral questions'}</p>
              </div>
            </div>
            {!status.questionnaire && <ChevronRight className="w-6 h-6 text-gray-300" />}
          </button>
        </div>

        {/* Generate Report Button */}
        <div className="pt-4 border-t border-gray-200">
          <button
            onClick={() => isReady ? onNavigate('results') : null}
            className={`w-full py-4 rounded-full font-bold text-lg transition-all shadow-md ${isReady ? 'bg-[#1A9E7A] hover:bg-teal-700 text-white shadow-teal-200' : 'bg-gray-200 text-gray-400 cursor-not-allowed shadow-none'}`}
            disabled={!isReady}
          >
            {isReady ? 'Generate Final Report' : 'Complete modules to unlock'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------
// 2.6 VIDEO ANALYSIS SCREEN
// ---------------------------------------------------------
function VideoAnalysisScreen({ onNavigate, onComplete, sessionUUID }) {
  const [recording, setRecording] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');
  const [error, setError] = useState('');

  // Fake progress for recording (camera not implemented)
  useEffect(() => {
    if (recording) {
      const interval = setInterval(() => {
        setProgress(p => {
          if (p >= 100) { clearInterval(interval); setRecording(false); return 100; }
          return p + 2;
        });
      }, 100);
      return () => clearInterval(interval);
    }
  }, [recording]);

  const handleFinish = () => {
    onComplete();
    onNavigate('assessment_hub');
  };

  const fileInputRef = useRef(null);

  const handleFileUpload = async (e) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setUploading(true); setError(''); setProgress(20); setStatusMsg('Uploading video...');

    try {
      // Step 1: Upload the file
      const formData = new FormData();
      formData.append('file', file);
      formData.append('session_uuid', sessionUUID);
      const uploadRes = await fetch('/api/v1/analyze/video/upload', { method: 'POST', body: formData });
      if (!uploadRes.ok) {
        const d = await uploadRes.json().catch(() => ({}));
        throw new Error(d.detail || 'Upload failed');
      }
      setProgress(40); setStatusMsg('Starting analysis...');

      // Step 2: Start analysis
      const startRes = await fetch('/api/v1/analyze/video/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_uuid: sessionUUID }),
      });
      if (!startRes.ok) {
        const d = await startRes.json().catch(() => ({}));
        throw new Error(d.detail || 'Could not start analysis');
      }
      setUploading(false); setAnalyzing(true); setProgress(60); setStatusMsg('Analyzing behavior...');

      // Step 3: Poll status every 2s
      let attempts = 0;
      const maxAttempts = 100;
      const poll = setInterval(async () => {
        attempts++;
        try {
          const statusRes = await fetch(`/api/v1/analyze/video/status/${sessionUUID}`);
          const statusData = await statusRes.json();

          if (statusData.status === 'video_done') {
            clearInterval(poll);
            setProgress(100); setAnalyzing(false); setStatusMsg('Analysis complete!');
          } else if (statusData.status === 'error_video') {
            clearInterval(poll);
            throw new Error(statusData.error || 'Video analysis failed');
          } else {
            setProgress(Math.min(90, 60 + attempts));
          }
        } catch (err) {
          clearInterval(poll);
          setError(err.message); setAnalyzing(false);
        }
        if (attempts >= maxAttempts) {
          clearInterval(poll);
          setError('Analysis timed out. Please try again.'); setAnalyzing(false);
        }
      }, 3000);
    } catch (err) {
      setError(err.message); setUploading(false); setAnalyzing(false);
    }
  };

  return (
    <div className="flex-1 bg-black flex flex-col h-full overflow-hidden relative">
      <div className="px-6 py-5 bg-gradient-to-b from-black/60 to-transparent flex items-center gap-4 z-20 absolute top-0 left-0 w-full">
        <button onClick={() => onNavigate('assessment_hub')} className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center text-white backdrop-blur-md">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <span className="text-white font-bold tracking-widest text-sm uppercase">Camera</span>
      </div>
      
      {/* Mock Camera Viewfinder */}
      <div className="flex-1 flex items-center justify-center relative">
        <div className="absolute inset-0 bg-gray-900 flex items-center justify-center">
          <User className="w-32 h-32 text-gray-700 opacity-50" />
        </div>
        
        {/* Status Overlay */}
        {(recording || uploading || analyzing) && (
          <div className="absolute top-24 right-6 flex items-center gap-2 bg-black/50 backdrop-blur-md px-3 py-1.5 rounded-full z-20">
            <div className={`w-2.5 h-2.5 rounded-full animate-pulse ${uploading || analyzing ? 'bg-blue-500' : 'bg-red-500'}`}></div>
            <span className={`font-mono text-sm font-bold ${uploading || analyzing ? 'text-blue-500' : 'text-red-500'}`}>
              {statusMsg || (uploading ? 'UPLOADING' : analyzing ? 'ANALYZING' : 'REC')}
            </span>
          </div>
        )}

        {/* Error overlay */}
        {error && (
          <div className="absolute top-24 left-6 right-6 bg-red-500/90 text-white px-4 py-3 rounded-xl z-20 text-sm font-medium">
            {error}
          </div>
        )}

        {/* Alignment Guide */}
        <div className="w-64 h-80 border-2 border-dashed border-white/30 rounded-full absolute z-10 flex items-center justify-center pointer-events-none">
          {!recording && !uploading && !analyzing && progress === 0 && <span className="text-white/50 text-xs font-bold uppercase tracking-widest absolute bottom-4">Align Face Here</span>}
        </div>
      </div>

      {/* Controls Container */}
      <div className="bg-black pb-10 pt-6 px-6 relative z-20 md:max-w-4xl md:mx-auto md:w-full">
        {progress > 0 && progress < 100 && (
          <div className="w-full bg-gray-800 h-1.5 rounded-full mb-6 overflow-hidden">
            <div className={`h-full ${uploading || analyzing ? 'bg-blue-500' : 'bg-red-500'} transition-all duration-300`} style={{ width: `${progress}%` }}></div>
          </div>
        )}
        
        <div className="flex justify-center items-center gap-6">
          {progress === 100 ? (
            <button onClick={handleFinish} className="bg-[#2ECC71] text-white font-bold py-4 px-12 rounded-full shadow-[0_0_20px_rgba(46,204,113,0.4)] text-lg flex items-center gap-2 hover:bg-green-500 transition-colors">
              <Check className="w-6 h-6" /> Submit Video
            </button>
          ) : (
            <>
              <button 
                onClick={() => setRecording(true)} 
                disabled={recording || uploading || analyzing}
                className={`w-20 h-20 rounded-full border-4 border-white flex items-center justify-center transition-all ${recording ? 'scale-90 opacity-80' : 'hover:scale-105'} ${(recording || uploading || analyzing) ? 'opacity-50' : ''}`}
              >
                <div className={`w-16 h-16 rounded-full transition-all ${recording ? 'bg-red-500 rounded-lg scale-50' : 'bg-red-500'}`}></div>
              </button>
              
              {(!recording && !uploading && !analyzing) && (
                <div className="flex flex-col items-center">
                  <span className="text-gray-500 text-xs font-bold mb-2">OR</span>
                  <input type="file" accept="video/*" className="hidden" ref={fileInputRef} onChange={handleFileUpload} />
                  <button onClick={() => fileInputRef.current?.click()} className="bg-gray-800 hover:bg-gray-700 text-white px-6 py-3 rounded-full text-sm font-bold flex items-center gap-2 transition-all border border-gray-700">
                    <Video className="w-4 h-4" /> Upload File
                  </button>
                </div>
              )}
            </>
          )}
        </div>
        {!uploading && !recording && !analyzing && progress === 0 && <p className="text-center text-gray-500 text-xs mt-6">Ensure good lighting and capture child's natural play.</p>}
      </div>
    </div>
  );
}


// ---------------------------------------------------------
// 3. QUESTIONNAIRE SCREEN
// ---------------------------------------------------------
function QuestionnaireScreen({ onNavigate, onComplete, sessionUUID }) {
  const [expandedSection, setExpandedSection] = useState(0);
  const [answers, setAnswers] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const sections = [
    { title: "Social Interaction", count: 10 },
    { title: "Communication", count: 10 },
    { title: "Behavior Patterns", count: 10 },
    { title: "Sensory & Emotional", count: 10 }
  ];

  const questionsBank = [
    // Social Interaction
    [
      "Does your child respond to their name?",
      "Does your child make eye contact?",
      "Does your child smile back when smiled at?",
      "Does your child show interest in other children?",
      "Does your child try to share enjoyment with you?",
      "Does your child point to show interest?",
      "Does your child wave goodbye?",
      "Does your child imitate actions?",
      "Does your child follow your gaze?",
      "Does your child bring objects to show you?"
    ],
    // Communication
    [
      "Does your child use gestures to communicate?",
      "Does your child understand simple instructions?",
      "Does your child babble or speak words?",
      "Does your child respond when spoken to?",
      "Does your child use meaningful sounds?",
      "Does your child ask for help when needed?",
      "Does your child respond to emotions?",
      "Does your child engage in back-and-forth sounds?",
      "Does your child use facial expressions?",
      "Does your child try to start conversations?"
    ],
    // Behavior Patterns
    [
      "Does your child repeat actions again and again?",
      "Does your child show unusual attachment to objects?",
      "Does your child line up toys?",
      "Does your child get upset with small changes?",
      "Does your child show repetitive movements (e.g. hand flapping)?",
      "Does your child focus on parts of objects?",
      "Does your child spin or rock frequently?",
      "Does your child insist on routines?",
      "Does your child play with toys in unusual ways?",
      "Does your child show intense interest in specific things?"
    ],
    // Sensory & Emotional
    [
      "Does your child react strongly to loud sounds?",
      "Does your child avoid eye contact?",
      "Does your child show limited emotional expression?",
      "Does your child overreact to touch?",
      "Does your child ignore pain or temperature?",
      "Does your child get easily frustrated?",
      "Does your child have difficulty calming down?",
      "Does your child show fear without reason?",
      "Does your child avoid social interaction?",
      "Does your child prefer to play alone?"
    ]
  ];

  const toggleAnswer = (sIdx, qIdx, value) => {
    const key = `${sIdx}-${qIdx}`;
    setAnswers({ ...answers, [key]: value });
  };

  const answeredCount = Object.keys(answers).length;
  const progressPercent = Math.min(100, Math.round((answeredCount / 40) * 100));

  return (
    <div className="flex-1 bg-white flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-6 py-5 bg-white border-b border-gray-100 flex items-center gap-4 z-10 shadow-sm relative">
        <button onClick={() => onNavigate('assessment_hub')} className="w-10 h-10 rounded-full bg-gray-50 flex items-center justify-center text-gray-600 hover:bg-gray-100">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex-1">
          <h2 className="text-lg font-bold text-gray-900">Behavioral Questionnaire</h2>
          <p className="text-sm text-gray-500">{answeredCount} of 40 answered</p>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-gray-100 h-1.5">
        <div className="bg-[#1A9E7A] h-full transition-all duration-500 ease-in-out" style={{ width: `${progressPercent}%` }}></div>
      </div>

      <div className="flex-1 overflow-y-auto p-6 pb-28">
        <div className="space-y-4 md:max-w-4xl md:mx-auto mt-4">
          {sections.map((section, sIdx) => (
            <div key={sIdx} className="border border-gray-200 rounded-2xl overflow-hidden bg-gray-50/50">
              <button
                onClick={() => setExpandedSection(expandedSection === sIdx ? -1 : sIdx)}
                className="w-full px-5 py-4 flex items-center justify-between bg-white text-left font-bold text-gray-800"
              >
                <div className="flex items-center gap-3">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm ${expandedSection === sIdx ? 'bg-[#1A9E7A] text-white' : 'bg-gray-100 text-gray-500'}`}>
                    {sIdx + 1}
                  </div>
                  {section.title}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-gray-400 bg-gray-100 px-2 py-1 rounded-md">{section.count} Qs</span>
                  <ChevronRight className={`w-5 h-5 text-gray-400 transition-transform ${expandedSection === sIdx ? 'rotate-90' : ''}`} />
                </div>
              </button>

              {expandedSection === sIdx && (
                <div className="px-5 py-4 space-y-6">
                  {questionsBank[sIdx].map((q, qIdx) => {
                    const key = `${sIdx}-${qIdx}`;
                    return (
                      <div key={qIdx} className="space-y-3">
                        <p className="text-gray-800 font-medium leading-snug text-[15px]">{qIdx + 1}. {q}</p>
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                          {[
                            { val: 0, label: 'Never' },
                            { val: 1, label: 'Rarely' },
                            { val: 2, label: 'Sometimes' },
                            { val: 3, label: 'Often' },
                            { val: 4, label: 'Always' }
                          ].map(opt => {
                            const isSelected = answers[key] === opt.val;
                            return (
                              <button
                                key={opt.val}
                                onClick={() => toggleAnswer(sIdx, qIdx, opt.val)}
                                className={`flex flex-col items-center justify-center p-2 rounded-xl border transition-all ${isSelected
                                    ? 'bg-[#1A9E7A] border-[#1A9E7A] text-white shadow-md'
                                    : 'bg-white border-gray-200 text-gray-600 hover:border-[#1A9E7A] hover:bg-teal-50'
                                  }`}
                              >
                                <span className="font-bold">{opt.val}</span>
                                <span className="text-[10px] uppercase tracking-wide font-semibold mt-0.5">{opt.label}</span>
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Bottom Action Area */}
      <div className="absolute bottom-0 left-0 w-full bg-white border-t border-gray-100 p-6 pt-4 pb-8 shadow-[0_-10px_30px_rgba(0,0,0,0.05)] md:max-w-4xl md:left-1/2 md:-translate-x-1/2 z-20">
        {error && (
          <div className="w-full bg-red-50 border border-red-200 text-red-600 text-sm px-4 py-2 rounded-xl mb-3 font-medium">{error}</div>
        )}
        <button
          onClick={async () => {
            if (answeredCount < 40) { setError('Please answer all 40 questions'); return; }
            setSubmitting(true); setError('');
            try {
              // Flatten answers object to ordered array of 40 ints
              const responses = [];
              for (let s = 0; s < 4; s++) {
                for (let q = 0; q < 10; q++) {
                  responses.push(answers[`${s}-${q}`] ?? 0);
                }
              }
              const res = await fetch('/api/v1/analyze/questionnaire', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  session_uuid: sessionUUID,
                  responses,
                  child_age_months: 36,
                  child_gender: 'unspecified',
                }),
              });
              if (!res.ok) {
                const d = await res.json().catch(() => ({}));
                throw new Error(d.detail || 'Submission failed');
              }
              onComplete();
              onNavigate('assessment_hub');
            } catch (err) {
              setError(err.message);
            } finally { setSubmitting(false); }
          }}
          disabled={submitting || answeredCount < 40}
          className={`w-full font-bold py-4 rounded-full shadow-lg transition-all text-lg ${answeredCount < 40 ? 'bg-gray-300 text-gray-500 cursor-not-allowed shadow-none' : 'bg-[#2ECC71] hover:bg-green-500 text-white shadow-green-200'} disabled:opacity-60`}
        >
          Submit Answers
        </button>
      </div>
    </div>
  );
}

function ResultsScreen({ onNavigate, sessionUUID }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!sessionUUID) { setError('No session found'); setLoading(false); return; }
    const loadReport = async () => {
      try {
        // Step 1: Run fusion
        await fetch('/api/v1/analyze/fuse', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_uuid: sessionUUID }),
        });
        // Step 2: Fetch full report
        const res = await fetch(`/api/v1/analyze/report/${sessionUUID}`);
        if (!res.ok) {
          const d = await res.json().catch(() => ({}));
          throw new Error(d.detail || 'Could not load report');
        }
        const data = await res.json();
        setReport(data);
      } catch (err) {
        setError(err.message);
      } finally { setLoading(false); }
    };
    loadReport();
  }, [sessionUUID]);

  const riskValue = report ? Math.round(report.final_risk_score * 100) : 0;
  const riskLevel = report?.risk_level || 'unknown';
  const confidencePercent = report ? Math.round(report.confidence * 100) : 0;

  const chartData = [
    { name: 'Risk', value: riskValue },
    { name: 'Safe', value: 100 - riskValue },
  ];
  const riskColors = { low: '#22c55e', medium: '#f59e0b', high: '#ef4444' };
  const COLORS = [riskColors[riskLevel] || '#6b7280', '#f3f4f6'];

  const categories = report?.category_scores ? Object.entries(report.category_scores).map(([label, val]) => {
    const v = typeof val === 'number' ? val : 0;
    return {
      label: label.charAt(0).toUpperCase() + label.slice(1),
      val: v,
      color: v >= 0.7 ? 'bg-red-500' : v >= 0.4 ? 'bg-orange-500' : 'bg-green-500',
    };
  }) : [];

  const riskBadge = {
    low: { bg: 'bg-green-50 border-green-100 text-green-600', text: 'LOW RISK' },
    medium: { bg: 'bg-amber-50 border-amber-100 text-amber-600', text: 'MEDIUM RISK' },
    high: { bg: 'bg-red-50 border-red-100 text-red-600', text: 'HIGH RISK' },
  }[riskLevel] || { bg: 'bg-gray-50 border-gray-200 text-gray-500', text: 'PENDING' };

  if (loading) {
    return (
      <div className="flex-1 bg-gray-50 flex flex-col items-center justify-center h-full">
        <div className="w-16 h-16 border-4 border-teal-200 border-t-[#1A9E7A] rounded-full animate-spin mb-4"></div>
        <p className="text-gray-500 font-medium">Generating your report...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 bg-gray-50 flex flex-col items-center justify-center h-full p-6">
        <AlertTriangle className="w-12 h-12 text-red-400 mb-4" />
        <p className="text-red-600 font-bold text-lg mb-2">Report Error</p>
        <p className="text-gray-500 text-center mb-6">{error}</p>
        <button onClick={() => onNavigate('home')} className="bg-[#1A9E7A] text-white px-6 py-3 rounded-full font-bold">Go Home</button>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-gray-50 flex flex-col h-full overflow-hidden">
      <div className="px-6 py-5 bg-white flex items-center gap-4 z-10 shadow-sm relative">
        <button onClick={() => onNavigate('home')} className="w-10 h-10 rounded-full bg-gray-50 flex items-center justify-center text-gray-600 hover:bg-gray-100">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <h2 className="text-lg font-bold text-gray-900">Assessment Results</h2>
      </div>

      <div className="flex-1 overflow-y-auto pb-24 md:pb-8 flex flex-col md:flex-row md:items-start md:p-8 md:gap-8">
        {/* Main Chart Card */}
        <div className="bg-white px-6 py-8 shadow-sm flex flex-col items-center border-b md:border md:rounded-3xl border-gray-100 md:w-[40%] flex-shrink-0">
          <div className="relative w-48 h-48 mb-2">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={chartData} cx="50%" cy="50%" innerRadius={70} outerRadius={90} startAngle={90} endAngle={-270} dataKey="value" stroke="none" cornerRadius={10}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-4xl font-black text-gray-900">{riskValue}%</span>
              <span className="text-xs text-gray-500 font-semibold uppercase tracking-wider mt-1">Probability</span>
            </div>
          </div>

          <div className={`${riskBadge.bg} border px-4 py-1.5 rounded-full font-bold text-sm flex items-center gap-2 mb-2`}>
            <AlertTriangle className="w-4 h-4" />
            {riskBadge.text}
          </div>
          <p className="text-gray-500 text-sm font-medium">Confidence Score: <span className="text-gray-900 font-bold">{confidencePercent}%</span></p>

          {report?.video_fallback_used && (
            <p className="text-xs text-amber-600 mt-2 font-medium">⚠ Video not submitted — results based on questionnaire only</p>
          )}
        </div>

        <div className="md:flex-1 md:flex md:flex-col md:w-full">
          {/* Disclaimer */}
          <div className="p-6 md:p-0 pb-2 md:pb-5">
            <div className="bg-amber-50 border border-amber-200 rounded-2xl p-4 flex gap-3 text-amber-800">
            <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
            <p className="text-sm font-medium leading-relaxed">
              {report?.disclaimer || 'This is a risk assessment, NOT a clinical diagnosis. Please share these results with a qualified healthcare professional.'}
            </p>
          </div>
        </div>

        {/* Breakdown */}
        {categories.length > 0 && (
        <div className="px-6 md:px-0 py-4">
          <h3 className="font-bold text-gray-900 mb-4 text-lg md:text-xl">Detailed Breakdown</h3>
          <div className="space-y-5 bg-white p-5 md:p-8 rounded-3xl shadow-sm border border-gray-100">
            {categories.map((cat, i) => (
              <div key={i}>
                <div className="flex justify-between text-sm mb-1.5 font-medium">
                  <span className="text-gray-700">{cat.label}</span>
                  <span className="text-gray-900 font-bold">{(cat.val * 10).toFixed(1)}/10</span>
                </div>
                <div className="h-2.5 w-full bg-gray-100 rounded-full overflow-hidden">
                  <div className={`h-full ${cat.color} rounded-full`} style={{ width: `${cat.val * 100}%` }}></div>
                </div>
              </div>
            ))}
          </div>
        </div>
        )}

        {/* Contribution info */}
        {report && (
        <div className="px-6 md:px-0 py-2">
          <div className="bg-white rounded-2xl p-4 shadow-sm border border-gray-100 space-y-2">
            <p className="text-sm text-gray-600"><span className="font-bold">Video:</span> {report.video_contribution || 'N/A'}</p>
            <p className="text-sm text-gray-600"><span className="font-bold">Questionnaire:</span> {report.questionnaire_contribution || 'N/A'}</p>
          </div>
        </div>
        )}
        </div>
      </div>

      {/* Bottom Actions */}
      <div className="absolute bottom-0 left-0 w-full bg-white p-4 pb-8 md:pb-6 flex flex-col gap-2 border-t border-gray-100 shadow-[0_-10px_20px_rgba(0,0,0,0.03)] z-20 md:max-w-4xl md:left-1/2 md:-translate-x-1/2">
        <p className="text-center text-xs text-gray-500 mb-1 font-medium">Have questions about this report?</p>
        <button
          onClick={() => onNavigate('chatbot')}
          className="w-full bg-[#1A9E7A] text-white font-bold py-4 rounded-2xl shadow-lg shadow-teal-200 flex items-center justify-center gap-2 hover:bg-teal-700 transition-all text-lg"
        >
          <Brain className="w-6 h-6" />
          <span>Consult AI Specialist Chatbot</span>
        </button>
      </div>
    </div>
  );
}



// ---------------------------------------------------------
// 5. AI CHATBOT SCREEN
// ---------------------------------------------------------
function ChatbotScreen({ onNavigate }) {
  const [messages, setMessages] = useState([
    { id: 1, sender: 'ai', text: "Based on the assessment, I noticed some patterns worth discussing. Would you like to understand what 'low eye contact' means for your development phase?", time: "10:02 AM" }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const endRef = useRef(null);

  const handleSend = () => {
    if (!inputText.trim()) return;

    const newMsg = { id: Date.now(), sender: 'user', text: inputText, time: "10:05 AM" };
    setMessages([...messages, newMsg]);
    setInputText('');
    setIsTyping(true);

    // Mock reply
    setTimeout(() => {
      setIsTyping(false);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        sender: 'ai',
        text: "That's a great question. Reduced eye contact can be an early indicator of neurodivergence, but every child develops differently. It might be helpful to discuss this specific behavior with a local pediatrician.",
        time: "10:05 AM"
      }]);
    }, 1500);
  };

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  return (
    <div className="flex-1 bg-gray-50 flex flex-col h-full">
      <div className="px-5 py-4 bg-white flex items-center gap-4 border-b border-gray-100 shadow-sm relative z-10">
        <button onClick={() => onNavigate('results')} className="w-10 h-10 rounded-full bg-gray-50 flex items-center justify-center text-gray-600">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex-1 flex flex-col justify-center">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-bold text-gray-900">AutiAssist AI</h2>
            <div className="bg-teal-100 text-[#1A9E7A] text-[10px] uppercase font-black px-1.5 py-0.5 rounded">BETA</div>
          </div>
          <span className="text-xs text-[#2ECC71] font-medium flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-[#2ECC71]"></span> Online
          </span>
        </div>
        <div className="w-10 h-10 rounded-full bg-teal-50 flex items-center justify-center">
          <Brain className="w-5 h-5 text-[#1A9E7A]" />
        </div>
      </div>

      <div className="bg-amber-50 py-2 px-4 text-center text-xs text-amber-800 border-b border-amber-100 shadow-sm">
        <span className="font-semibold">Disclaimer:</span> This AI does not provide medical advice.
      </div>

      <div className="flex-1 overflow-y-auto p-5 md:p-8 space-y-4 md:space-y-6 md:max-w-4xl md:mx-auto md:w-full">
        {messages.map(msg => (
          <div key={msg.id} className={`flex flex-col max-w-[85%] ${msg.sender === 'user' ? 'ml-auto items-end' : 'mr-auto items-start'}`}>
            <div className={`px-4 py-3 rounded-2xl shadow-sm text-[15px] leading-relaxed ${msg.sender === 'user'
                ? 'bg-[#1A9E7A] text-white rounded-tr-sm'
                : 'bg-white border border-gray-100 text-gray-800 rounded-tl-sm'
              }`}>
              {msg.text}
            </div>
            <span className="text-[11px] text-gray-400 mt-1 font-medium px-1">{msg.time}</span>
          </div>
        ))}
        {isTyping && (
          <div className="bg-white border border-gray-100 px-4 py-3.5 rounded-2xl rounded-tl-sm shadow-sm w-fit flex items-center gap-1.5">
            <span className="w-2 h-2 bg-teal-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
            <span className="w-2 h-2 bg-teal-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
            <span className="w-2 h-2 bg-teal-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="bg-white p-4 border-t border-gray-100 z-10 pt-3 pb-8 md:pb-6 shadow-[0_-10px_20px_rgba(0,0,0,0.02)]">
        <div className="flex items-center gap-2 bg-gray-50 p-1.5 rounded-full border border-gray-200 focus-within:border-[#1A9E7A] focus-within:ring-2 focus-within:ring-teal-100 transition-all md:max-w-4xl md:mx-auto">
          <input
            type="text"
            placeholder="Type your question..."
            className="flex-1 bg-transparent border-none focus:outline-none px-4 py-2 text-sm text-gray-800"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          />
          <button
            onClick={handleSend}
            disabled={!inputText.trim()}
            className="w-10 h-10 rounded-full bg-[#1A9E7A] text-white flex items-center justify-center disabled:bg-gray-300 transition-colors shadow-sm"
          >
            <Send className="w-4 h-4 ml-0.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------
// 6. FIND CLINICS SCREEN
// ---------------------------------------------------------
function ClinicsScreen({ onNavigate }) {
  const clinics = [
    { name: "Child Neuro Center", tag: "Neurologist", color: "bg-blue-100 text-blue-700", distance: "2.3 km", rating: 4.8, address: "123 Medical Parkway" },
    { name: "Bright Steps Therapy", tag: "Therapy Center", color: "bg-orange-100 text-orange-700", distance: "4.1 km", rating: 4.9, address: "78 Wellness Ave" },
    { name: "Dr. Sarah Jenkins", tag: "Psychologist", color: "bg-purple-100 text-purple-700", distance: "5.5 km", rating: 4.6, address: "Suite 402, North Wing" },
  ];

  const filters = ["All", "Neurologist", "Psychologist", "Therapy"];

  return (
    <div className="flex-1 bg-gray-50 flex flex-col h-full overflow-hidden pb-16">
      <div className="bg-[#1A9E7A] pt-5 px-5 pb-6 rounded-b-3xl shadow-md z-10 relative">
        <div className="flex items-center justify-between gap-4 mb-5">
          <button onClick={() => onNavigate('results')} className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center text-white backdrop-blur-sm">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <h2 className="text-lg font-bold text-white">Nearby Specialists</h2>
          <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center text-white backdrop-blur-sm cursor-pointer">
            <MapPin className="w-5 h-5" />
          </div>
        </div>

        <div className="relative">
          <Search className="absolute left-4 top-3.5 w-5 h-5 text-teal-700" />
          <input
            type="text"
            placeholder="Search city or pincode"
            className="w-full bg-white rounded-2xl py-3.5 pl-12 pr-4 text-gray-800 font-medium focus:outline-none shadow-sm placeholder-teal-800/50"
            defaultValue="New York, NY 10001"
          />
        </div>
      </div>

      <div className="overflow-x-auto py-4 px-5 scrollbar-hide flex gap-2 snap-x">
        {filters.map((f, i) => (
          <button key={i} className={`whitespace-nowrap px-4 py-2 rounded-full text-sm font-bold snap-start ${i === 0 ? 'bg-gray-800 text-white shadow-sm' : 'bg-white text-gray-500 border border-gray-200 hover:bg-gray-50'}`}>
            {f}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto px-5 pb-6 space-y-4 md:space-y-0 md:grid md:grid-cols-2 lg:grid-cols-3 md:gap-5">
        {clinics.map((clinic, i) => (
          <div key={i} className="bg-white rounded-3xl p-5 shadow-sm border border-gray-100 relative overflow-hidden flex flex-col h-full">
            <div className="flex justify-between items-start mb-2">
              <h3 className="font-bold text-gray-900 text-[17px]">{clinic.name}</h3>
              <div className="flex items-center gap-1 bg-yellow-50 px-2 py-1 rounded-md text-amber-600 font-bold text-xs border border-amber-100">
                <Star className="w-3 h-3 fill-currentColor" />
                {clinic.rating}
              </div>
            </div>

            <span className={`inline-block px-3 py-1 rounded-md text-xs font-bold mb-3 ${clinic.color}`}>
              {clinic.tag}
            </span>

            <p className="text-gray-500 text-sm mb-4 flex items-center gap-2">
              <MapPin className="w-4 h-4 text-gray-400 shrink-0" />
              {clinic.address}
            </p>

            <div className="flex items-center justify-between pt-4 mt-auto border-t border-gray-100">
              <span className="font-semibold text-gray-900 text-sm">{clinic.distance} away</span>
              <button className="text-[#1A9E7A] font-bold text-sm bg-teal-50 px-4 py-2 rounded-full hover:bg-teal-100 transition-colors">
                Get Directions
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------
// COMMON BOTTOM TAB BAR
// ---------------------------------------------------------
function BottomTabBar({ current, onNavigate }) {
  return (
    <div className="absolute md:static bottom-0 left-0 w-full md:w-64 bg-white md:bg-gray-50 h-20 md:h-screen flex md:flex-col justify-around md:justify-start items-center md:items-stretch px-6 md:px-4 md:py-8 rounded-t-3xl md:rounded-none shadow-[0_-10px_30px_rgba(0,0,0,0.06)] md:shadow-none border-t md:border-t-0 md:border-r border-gray-100 z-50 transition-all flex-shrink-0">
      
      {/* Desktop Logo Header */}
      <div className="hidden md:flex items-center gap-3 mb-10 px-4">
        <div className="w-10 h-10 bg-white shadow-sm rounded-xl flex items-center justify-center relative">
          <Brain className="w-6 h-6 text-[#1A9E7A]" />
          <Heart className="w-3 h-3 text-[#2ECC71] absolute right-1 bottom-1" fill="currentColor" />
        </div>
        <h1 className="text-xl font-bold text-gray-900 tracking-tight">AutiSense</h1>
      </div>

      <div className="flex flex-row md:flex-col justify-around md:justify-start w-full md:gap-3">
        <button
          onClick={() => onNavigate('home')}
          className={`flex flex-col md:flex-row items-center md:items-start gap-1.5 md:gap-4 w-16 md:w-full md:px-4 md:py-3.5 md:rounded-2xl transition-all ${current === 'assessment' ? `${PrimaryColor} md:bg-white md:shadow-sm md:border md:border-gray-100` : 'text-gray-400 hover:text-gray-600 md:hover:bg-gray-100'}`}
        >
          <div className={`p-1.5 md:p-0 rounded-xl ${current === 'assessment' ? 'bg-teal-50 md:bg-transparent text-[#1A9E7A]' : ''}`}>
            <Home className={`w-6 h-6 md:w-5 md:h-5 ${current === 'assessment' ? 'fill-[#1A9E7A]/20' : ''}`} />
          </div>
          <span className="text-[10px] md:text-[14px] font-bold tracking-wide">Home</span>
        </button>

        <button
          onClick={() => onNavigate('history')}
          className={`flex flex-col md:flex-row items-center md:items-start gap-1.5 md:gap-4 w-16 md:w-full md:px-4 md:py-3.5 md:rounded-2xl transition-all ${current === 'history' ? `${PrimaryColor} md:bg-white md:shadow-sm md:border md:border-gray-100` : 'text-gray-400 hover:text-gray-600 md:hover:bg-gray-100'}`}
        >
          <div className={`p-1.5 md:p-0 rounded-xl ${current === 'history' ? 'bg-teal-50 md:bg-transparent text-[#1A9E7A]' : ''}`}>
            <ClipboardList className={`w-6 h-6 md:w-5 md:h-5 ${current === 'history' ? 'fill-[#1A9E7A]/20' : ''}`} />
          </div>
          <span className="text-[10px] md:text-[14px] font-bold tracking-wide">History</span>
        </button>

        <button
          onClick={() => onNavigate('clinics')}
          className={`flex flex-col md:flex-row items-center md:items-start gap-1.5 md:gap-4 w-16 md:w-full md:px-4 md:py-3.5 md:rounded-2xl transition-all ${current === 'clinics' ? `${PrimaryColor} md:bg-white md:shadow-sm md:border md:border-gray-100` : 'text-gray-400 hover:text-gray-600 md:hover:bg-gray-100'}`}
        >
          <div className={`p-1.5 md:p-0 rounded-xl ${current === 'clinics' ? 'bg-teal-50 md:bg-transparent text-[#1A9E7A]' : ''}`}>
            <MapPin className={`w-6 h-6 md:w-5 md:h-5 ${current === 'clinics' ? 'fill-[#1A9E7A]/20' : ''}`} />
          </div>
          <span className="text-[10px] md:text-[14px] font-bold tracking-wide">Clinics</span>
        </button>

        <button
          onClick={() => onNavigate('profile')}
          className={`flex flex-col md:flex-row items-center md:items-start gap-1.5 md:gap-4 w-16 md:w-full md:px-4 md:py-3.5 md:rounded-2xl transition-all ${current === 'profile' ? `${PrimaryColor} md:bg-white md:shadow-sm md:border md:border-gray-100` : 'text-gray-400 hover:text-gray-600 md:hover:bg-gray-100'}`}
        >
          <div className={`p-1.5 md:p-0 rounded-xl ${current === 'profile' ? 'bg-teal-50 md:bg-transparent text-[#1A9E7A]' : ''}`}>
            <Settings className={`w-6 h-6 md:w-5 md:h-5 ${current === 'profile' ? 'fill-[#1A9E7A]/20' : ''}`} />
          </div>
          <span className="text-[10px] md:text-[14px] font-bold tracking-wide">Profile</span>
        </button>
      </div>

      {/* Desktop Logout Button */}
      <div className="hidden md:block mt-auto w-full border-t border-gray-200 pt-6">
        <button onClick={() => onNavigate('login')} className="flex items-center gap-3 w-full px-4 py-3 text-red-500 font-bold hover:bg-red-50 rounded-2xl transition-colors">
          <ArrowLeft className="w-5 h-5" />
          Log Out
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------
// 7. CREATE ACCOUNT SCREEN
// ---------------------------------------------------------
function CreateAccountScreen({ onNavigate, createGuestSession, setAuthToken }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleRegister = async () => {
    if (!email || !password) { setError('Please fill all fields'); return; }
    setLoading(true); setError('');
    try {
      const res = await fetch('/api/v1/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || 'Registration failed');
      }
      const data = await res.json();
      setAuthToken(data.access_token);
      await createGuestSession();
      onNavigate('home');
    } catch (err) {
      setError(err.message);
    } finally { setLoading(false); }
  };

  return (
    <div className="flex-1 bg-gradient-to-br from-teal-50 to-white relative flex flex-col overflow-y-auto overflow-x-hidden w-full h-full">
      <div className="fixed top-0 left-0 w-full h-1/2 md:w-1/2 md:h-full bg-teal-100/40 rounded-b-[100%] md:rounded-b-none md:rounded-r-[100%] scale-150 -translate-y-1/4 md:translate-y-0 md:scale-110 md:-translate-x-10 origin-top md:origin-left pointer-events-none z-0"></div>

      <div className="absolute top-0 left-0 w-full p-6 flex items-center z-20">
        <button onClick={() => onNavigate('login')} className="w-12 h-12 rounded-full bg-white shadow-md flex items-center justify-center text-gray-600 hover:bg-gray-50 hover:scale-105 transition-all relative">
          <ArrowLeft className="w-6 h-6" />
        </button>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center p-6 min-h-[600px] w-full max-w-7xl mx-auto relative z-10 pt-24">
        <div className="w-full max-w-sm md:max-w-md bg-white/60 backdrop-blur-xl p-8 md:p-10 rounded-[2rem] shadow-xl border border-white/50 flex flex-col items-center text-center">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-2">Join AutiSense</h1>
          <p className="text-gray-500 font-medium mb-10 text-center">Start your child's journey today.</p>
          
          {error && (
            <div className="w-full bg-red-50 border border-red-200 text-red-600 text-sm px-4 py-3 rounded-xl mb-4 font-medium">{error}</div>
          )}

          <div className="w-full space-y-4 mb-8">
            <input type="text" placeholder="Full Name" value={name} onChange={e => setName(e.target.value)} className="w-full bg-white/80 border border-gray-200 rounded-2xl py-3.5 px-4 text-gray-700 focus:outline-none focus:ring-2 focus:ring-[#1A9E7A] focus:bg-white transition-all shadow-sm" />
            <input type="email" placeholder="Email address" value={email} onChange={e => setEmail(e.target.value)} className="w-full bg-white/80 border border-gray-200 rounded-2xl py-3.5 px-4 text-gray-700 focus:outline-none focus:ring-2 focus:ring-[#1A9E7A] focus:bg-white transition-all shadow-sm" />
            <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} className="w-full bg-white/80 border border-gray-200 rounded-2xl py-3.5 px-4 text-gray-700 focus:outline-none focus:ring-2 focus:ring-[#1A9E7A] focus:bg-white transition-all shadow-sm" />
          </div>
          
          <button onClick={handleRegister} disabled={loading} className="w-full bg-[#1A9E7A] hover:bg-teal-700 text-white font-bold py-4 rounded-full shadow-lg shadow-teal-200 hover:shadow-xl transition-all text-lg mb-4 disabled:opacity-50">
            {loading ? 'Creating account...' : 'Register'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------
// 8. HISTORY SCREEN
// ---------------------------------------------------------
function HistoryScreen({ onNavigate }) {
  const pastAssessments = [
    { date: "Oct 12, 2023", score: "68%", level: "Medium Risk" },
    { date: "Sep 05, 2023", score: "74%", level: "High Risk" },
    { date: "Aug 20, 2023", score: "70%", level: "High Risk" },
    { date: "Jul 11, 2023", score: "71%", level: "High Risk" }
  ];

  return (
    <div className="flex-1 bg-gray-50 flex flex-col h-full overflow-hidden pb-20">
      <div className="px-6 py-5 bg-white border-b border-gray-100 shadow-sm z-10">
        <h2 className="text-xl font-bold text-gray-900">Assessment History</h2>
      </div>
      <div className="flex-1 overflow-y-auto p-6 space-y-4 md:max-w-4xl md:mx-auto md:w-full md:mt-4">
        {pastAssessments.map((a, i) => (
          <div key={i} className="bg-white rounded-2xl p-5 shadow-sm border border-gray-100 flex justify-between items-center transition-transform hover:scale-[1.02]">
            <div>
              <p className="font-bold text-gray-900 text-[17px] mb-1.5">{a.date}</p>
              <span className={`text-[11px] font-bold px-2 py-1 rounded-md ${a.level.includes("High") ? "bg-red-50 text-red-600 border border-red-100" : "bg-orange-50 text-orange-600 border border-orange-100"}`}>
                {a.level}
              </span>
            </div>
            <div className="text-right">
              <span className="text-2xl font-black text-gray-800">{a.score}</span>
              <p className="text-[10px] text-gray-400 font-bold uppercase tracking-widest mt-0.5">Risk</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------
// 9. PROFILE SCREEN
// ---------------------------------------------------------
function ProfileScreen({ onNavigate }) {
  const options = ["Account Details", "Child Preferences", "Notifications", "Privacy Policy", "Log Out"];
  return (
    <div className="flex-1 bg-gray-50 flex flex-col h-full overflow-hidden pb-20">
      <div className="px-6 py-5 bg-white border-b border-gray-100 shadow-sm z-10">
        <h2 className="text-xl font-bold text-gray-900">Settings & Profile</h2>
      </div>
      <div className="flex-1 overflow-y-auto p-6 md:p-10 md:max-w-3xl md:mx-auto md:w-full">
        <div className="bg-white rounded-3xl p-6 md:p-10 shadow-sm border border-gray-100 flex flex-col items-center mb-6 relative overflow-hidden">
          <div className="absolute top-0 w-full h-16 md:h-24 bg-teal-50"></div>
          <div className="w-24 h-24 md:w-32 md:h-32 bg-white border-4 border-white shadow-sm text-teal-700 rounded-full flex items-center justify-center text-3xl font-bold mb-4 z-10">
            <User className="w-10 h-10 md:w-16 md:h-16 text-teal-600" />
          </div>
          <h3 className="text-xl font-bold text-gray-900 z-10">Jane's Parent</h3>
          <p className="text-gray-500 text-sm z-10 font-medium mt-1">jane.doe@example.com</p>
        </div>
        
        <div className="space-y-3">
          {options.map((opt, i) => (
            <button key={i} onClick={() => opt === "Log Out" ? onNavigate("login") : null} className={`w-full bg-white p-4 rounded-2xl border border-gray-100 flex items-center justify-between shadow-sm transition-all ${opt === "Log Out" ? "text-red-500 font-bold hover:bg-red-50 mt-8" : "text-gray-700 font-semibold hover:border-teal-200 hover:shadow-md hover:bg-teal-50 group"}`}>
              {opt}
              {opt !== "Log Out" && <ChevronRight className="w-5 h-5 text-gray-300 group-hover:text-teal-500 transition-colors" />}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
