import React, { useState, useEffect, useRef } from 'react';
import {
  Activity, Brain, Heart, Search, MessageSquare, Video,
  ClipboardList, BarChart2, MapPin, ChevronRight, ArrowLeft,
  Send, User, Home, Calendar, Settings, AlertTriangle, Check, Star
} from 'lucide-react';
import { useAuth } from './hooks/useAuth';
import { AuthAPI, VideoAPI, QuestionnaireAPI, ResultsAPI, TokenManager } from './services/api';
import PostAnalysisResult from './components/PostAnalysisResult';
import BottomTabBar from './components/BottomTabBar';
import LocationSpecialistWidget from './components/LocationSpecialistWidget';

const PrimaryColor = "text-[#1A9E7A]";
const PrimaryBg = "bg-[#1A9E7A]";

export default function AppContent() {
  const { user, loading: authLoading, logout } = useAuth();
  const [currentScreen, setCurrentScreen] = useState('login');
  const [assessmentStatus, setAssessmentStatus] = useState({ video: false, questionnaire: false });
  const [sessionUUID, setSessionUUID] = useState(() => localStorage.getItem('session_uuid'));

  const navigateTo = (screen) => setCurrentScreen(screen);
  const markComplete = (task) => setAssessmentStatus(prev => ({ ...prev, [task]: true }));

  // Route guard
  useEffect(() => {
    if (!authLoading && !user && currentScreen !== 'login' && currentScreen !== 'create_account') {
      navigateTo('login');
    }
  }, [user, authLoading, currentScreen]);

  // Ensure session exists
  useEffect(() => {
    async function ensureSession() {
      if (user && !sessionUUID) {
        try {
          const session = await AuthAPI.createGuestSession();
          setSessionUUID(session.session_uuid);
          localStorage.setItem('session_uuid', session.session_uuid);
        } catch (err) {
          console.error('Session creation failed:', err);
        }
      }
    }
    if (!authLoading) ensureSession();
  }, [user, authLoading, sessionUUID]);

  const handleLogout = () => {
    logout();
    setSessionUUID(null);
    setAssessmentStatus({ video: false, questionnaire: false });
    navigateTo('login');
  };

  if (authLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="w-16 h-16 border-4 border-teal-200 border-t-[#1A9E7A] rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 flex justify-center font-sans" style={{ fontFamily: "'DM Sans', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Sora:wght@400;600;700&display=swap');
        h1, h2, h3, h4, h5, h6 { font-family: 'Sora', sans-serif; }
      `}</style>

      <div className="w-full max-w-full bg-white shadow-xl min-h-screen relative flex flex-col md:flex-row overflow-hidden">
        {['home', 'assessment_hub', 'results', 'clinics', 'history', 'profile'].includes(currentScreen) && (
          <BottomTabBar 
            current={currentScreen === 'home' || currentScreen === 'assessment_hub' ? 'assessment' : currentScreen} 
            onNavigate={navigateTo} 
            onLogout={handleLogout} 
          />
        )}

        <div className={`flex-1 relative h-screen overflow-hidden flex flex-col ${['home', 'assessment_hub', 'results', 'clinics', 'history', 'profile'].includes(currentScreen) ? 'md:w-[calc(100%-16rem)]' : 'w-full'}`}>
          {currentScreen === 'login' && <LoginScreen onNavigate={navigateTo} />}
          {currentScreen === 'home' && <HomeScreen onNavigate={navigateTo} />}
          {currentScreen === 'assessment_hub' && <AssessmentHubScreen onNavigate={navigateTo} status={assessmentStatus} />}
          {currentScreen === 'video_analysis' && <VideoAnalysisScreen onNavigate={navigateTo} onComplete={() => markComplete('video')} sessionUUID={sessionUUID} />}
          {currentScreen === 'questionnaire' && <QuestionnaireScreen onNavigate={navigateTo} onComplete={() => markComplete('questionnaire')} sessionUUID={sessionUUID} />}
          {currentScreen === 'results' && <ResultsScreen onNavigate={navigateTo} sessionUUID={sessionUUID} />}
          {currentScreen === 'chatbot' && <ChatbotScreen onNavigate={navigateTo} />}
          {currentScreen === 'clinics' && <ClinicsScreen onNavigate={navigateTo} />}
          {currentScreen === 'create_account' && <CreateAccountScreen onNavigate={navigateTo} />}
          {currentScreen === 'history' && <HistoryScreen onNavigate={navigateTo} />}
          {currentScreen === 'profile' && <ProfileScreen onNavigate={navigateTo} onLogout={handleLogout} />}
        </div>
      </div>
    </div>
  );
}

// ── LOGIN SCREEN ───────────────────────────────────────────────────
function LoginScreen({ onNavigate }) {
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLogin = async () => {
    if (!email || !password) { setError('Please enter email and password'); return; }
    setLoading(true); setError('');
    try {
      await login(email, password);
      onNavigate('home');
    } catch (err) {
      setError(err.message);
    } finally { setLoading(false); }
  };

  const handleGuest = async () => {
    setLoading(true); setError('');
    try {
      const session = await AuthAPI.createGuestSession();
      localStorage.setItem('session_uuid', session.session_uuid);
      onNavigate('home');
    } catch (err) {
      setError('Could not connect to server');
    } finally { setLoading(false); }
  };

  return (
    <div className="flex-1 bg-gradient-to-br from-teal-50 to-white relative flex flex-col overflow-y-auto w-full h-full">
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
          </div>

          <button onClick={handleLogin} disabled={loading} className="w-full bg-[#1A9E7A] hover:bg-teal-700 text-white font-bold py-4 rounded-full shadow-lg shadow-teal-200 hover:shadow-xl transition-all mb-4 text-lg disabled:opacity-50">
            {loading ? 'Connecting...' : 'Sign In'}
          </button>

          <button onClick={() => onNavigate('create_account')} className="w-full text-[#1A9E7A] font-bold py-3.5 rounded-full mb-2 hover:bg-teal-50 transition-colors border-2 border-transparent hover:border-teal-100">
            Create Account
          </button>
          <button onClick={handleGuest} disabled={loading} className="w-full text-gray-500 font-bold py-3.5 rounded-full hover:bg-gray-50 transition-all text-sm border-2 border-transparent disabled:opacity-50">
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

// ── HOME SCREEN ───────────────────────────────────────────────────
function HomeScreen({ onNavigate }) {
  const { user } = useAuth();
  const displayName = user?.display_name || 'Guest';
  const initials = displayName.split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase();

  return (
    <div className="flex-1 bg-gray-50 pb-20 overflow-y-auto overflow-x-hidden flex flex-col">
      <div className="bg-white px-6 py-5 border-b border-gray-100 shadow-sm z-10 relative">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-teal-100 rounded-full border-2 border-white shadow-sm flex items-center justify-center overflow-hidden">
              <span className="text-teal-700 font-bold text-lg">{initials}</span>
            </div>
            <div>
              <p className="text-sm text-gray-500 font-medium">Welcome back,</p>
              <h2 className="text-xl font-bold text-gray-900">{displayName}</h2>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 p-6 flex flex-col items-center justify-center">
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
          <button onClick={() => onNavigate('assessment_hub')} className="bg-[#2ECC71] hover:bg-green-400 text-white font-bold py-4 px-6 rounded-full w-full shadow-lg transition-all relative z-10 text-lg flex items-center justify-center gap-2">
            Start Test <ChevronRight className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}

// ── ASSESSMENT HUB SCREEN ─────────────────────────────────────────
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
          <button onClick={() => onNavigate('video_analysis')} className={`w-full bg-white p-5 rounded-3xl shadow-sm border-2 text-left flex items-center justify-between transition-all ${status.video ? 'border-green-400 bg-green-50/30' : 'border-transparent hover:border-teal-200 hover:shadow-md'}`}>
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

          <button onClick={() => onNavigate('questionnaire')} className={`w-full bg-white p-5 rounded-3xl shadow-sm border-2 text-left flex items-center justify-between transition-all ${status.questionnaire ? 'border-green-400 bg-green-50/30' : 'border-transparent hover:border-teal-200 hover:shadow-md'}`}>
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

        <div className="pt-4 border-t border-gray-200">
          <button onClick={() => isReady ? onNavigate('results') : null} className={`w-full py-4 rounded-full font-bold text-lg transition-all shadow-md ${isReady ? 'bg-[#1A9E7A] hover:bg-teal-700 text-white shadow-teal-200' : 'bg-gray-200 text-gray-400 cursor-not-allowed shadow-none'}`} disabled={!isReady}>
            {isReady ? 'Generate Final Report' : 'Complete modules to unlock'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── VIDEO ANALYSIS SCREEN ─────────────────────────────────────────
function VideoAnalysisScreen({ onNavigate, onComplete, sessionUUID }) {
  const [recording, setRecording] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

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

  const handleFileUpload = async (e) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setUploading(true); setError(''); setProgress(20); setStatusMsg('Uploading video...');

    try {
      await VideoAPI.upload(sessionUUID, file);
      setProgress(40); setStatusMsg('Starting analysis...');

      await VideoAPI.startAnalysis(sessionUUID);
      setUploading(false); setAnalyzing(true); setProgress(60); setStatusMsg('Analyzing behavior...');

      let attempts = 0;
      const maxAttempts = 100;
      const poll = setInterval(async () => {
        attempts++;
        try {
          const statusData = await VideoAPI.getStatus(sessionUUID);
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
      
      <div className="flex-1 flex items-center justify-center relative">
        <div className="absolute inset-0 bg-gray-900 flex items-center justify-center">
          <User className="w-32 h-32 text-gray-700 opacity-50" />
        </div>
        
        {(recording || uploading || analyzing) && (
          <div className="absolute top-24 right-6 flex items-center gap-2 bg-black/50 backdrop-blur-md px-3 py-1.5 rounded-full z-20">
            <div className={`w-2.5 h-2.5 rounded-full animate-pulse ${uploading || analyzing ? 'bg-blue-500' : 'bg-red-500'}`}></div>
            <span className={`font-mono text-sm font-bold ${uploading || analyzing ? 'text-blue-500' : 'text-red-500'}`}>
              {statusMsg || (uploading ? 'UPLOADING' : analyzing ? 'ANALYZING' : 'REC')}
            </span>
          </div>
        )}

        {error && (
          <div className="absolute top-24 left-6 right-6 bg-red-500/90 text-white px-4 py-3 rounded-xl z-20 text-sm font-medium">
            {error}
          </div>
        )}

        <div className="w-64 h-80 border-2 border-dashed border-white/30 rounded-full absolute z-10 flex items-center justify-center pointer-events-none">
          {!recording && !uploading && !analyzing && progress === 0 && <span className="text-white/50 text-xs font-bold uppercase tracking-widest absolute bottom-4">Align Face Here</span>}
        </div>
      </div>

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
              <button onClick={() => setRecording(true)} disabled={recording || uploading || analyzing} className={`w-20 h-20 rounded-full border-4 border-white flex items-center justify-center transition-all ${recording ? 'scale-90 opacity-80' : 'hover:scale-105'} ${(recording || uploading || analyzing) ? 'opacity-50' : ''}`}>
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

// ── QUESTIONNAIRE SCREEN ─────────────────────────────────────────
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
    ["Does your child respond to their name?", "Does your child make eye contact?", "Does your child smile back when smiled at?", "Does your child show interest in other children?", "Does your child try to share enjoyment with you?", "Does your child point to show interest?", "Does your child wave goodbye?", "Does your child imitate actions?", "Does your child follow your gaze?", "Does your child bring objects to show you?"],
    ["Does your child use gestures to communicate?", "Does your child understand simple instructions?", "Does your child babble or speak words?", "Does your child respond when spoken to?", "Does your child use meaningful sounds?", "Does your child ask for help when needed?", "Does your child respond to emotions?", "Does your child engage in back-and-forth sounds?", "Does your child use facial expressions?", "Does your child try to start conversations?"],
    ["Does your child repeat actions again and again?", "Does your child show unusual attachment to objects?", "Does your child line up toys?", "Does your child get upset with small changes?", "Does your child show repetitive movements (e.g. hand flapping)?", "Does your child focus on parts of objects?", "Does your child spin or rock frequently?", "Does your child insist on routines?", "Does your child play with toys in unusual ways?", "Does your child show intense interest in specific things?"],
    ["Does your child react strongly to loud sounds?", "Does your child avoid eye contact?", "Does your child show limited emotional expression?", "Does your child overreact to touch?", "Does your child ignore pain or temperature?", "Does your child get easily frustrated?", "Does your child have difficulty calming down?", "Does your child show fear without reason?", "Does your child avoid social interaction?", "Does your child prefer to play alone?"]
  ];

  const toggleAnswer = (sIdx, qIdx, value) => {
    const key = `${sIdx}-${qIdx}`;
    setAnswers({ ...answers, [key]: value });
  };

  const answeredCount = Object.keys(answers).length;
  const progressPercent = Math.min(100, Math.round((answeredCount / 40) * 100));

  const handleSubmit = async () => {
    if (answeredCount < 40) { setError('Please answer all 40 questions'); return; }
    setSubmitting(true); setError('');
    try {
      const responses = [];
      for (let s = 0; s < 4; s++) {
        for (let q = 0; q < 10; q++) {
          responses.push(answers[`${s}-${q}`] ?? 0);
        }
      }
      await QuestionnaireAPI.submit(sessionUUID, responses);
      onComplete();
      onNavigate('assessment_hub');
    } catch (err) {
      setError(err.message);
    } finally { setSubmitting(false); }
  };

  return (
    <div className="flex-1 bg-white flex flex-col h-full overflow-hidden">
      <div className="px-6 py-5 bg-white border-b border-gray-100 flex items-center gap-4 z-10 shadow-sm relative">
        <button onClick={() => onNavigate('assessment_hub')} className="w-10 h-10 rounded-full bg-gray-50 flex items-center justify-center text-gray-600 hover:bg-gray-100">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex-1">
          <h2 className="text-lg font-bold text-gray-900">Behavioral Questionnaire</h2>
          <p className="text-sm text-gray-500">{answeredCount} of 40 answered</p>
        </div>
      </div>

      <div className="w-full bg-gray-100 h-1.5">
        <div className="bg-[#1A9E7A] h-full transition-all duration-500 ease-in-out" style={{ width: `${progressPercent}%` }}></div>
      </div>

      <div className="flex-1 overflow-y-auto p-6 pb-28">
        <div className="space-y-4 md:max-w-4xl md:mx-auto mt-4">
          {sections.map((section, sIdx) => (
            <div key={sIdx} className="border border-gray-200 rounded-2xl overflow-hidden bg-gray-50/50">
              <button onClick={() => setExpandedSection(expandedSection === sIdx ? -1 : sIdx)} className="w-full px-5 py-4 flex items-center justify-between bg-white text-left font-bold text-gray-800">
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
                          {[{ val: 0, label: 'Never' }, { val: 1, label: 'Rarely' }, { val: 2, label: 'Sometimes' }, { val: 3, label: 'Often' }, { val: 4, label: 'Always' }].map(opt => {
                            const isSelected = answers[key] === opt.val;
                            return (
                              <button key={opt.val} onClick={() => toggleAnswer(sIdx, qIdx, opt.val)} className={`flex flex-col items-center justify-center p-2 rounded-xl border transition-all ${isSelected ? 'bg-[#1A9E7A] border-[#1A9E7A] text-white shadow-md' : 'bg-white border-gray-200 text-gray-600 hover:border-[#1A9E7A] hover:bg-teal-50'}`}>
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

      <div className="absolute bottom-0 left-0 w-full bg-white border-t border-gray-100 p-6 pt-4 pb-8 shadow-[0_-10px_30px_rgba(0,0,0,0.05)] md:max-w-4xl md:left-1/2 md:-translate-x-1/2 z-20">
        {error && <div className="w-full bg-red-50 border border-red-200 text-red-600 text-sm px-4 py-2 rounded-xl mb-3 font-medium">{error}</div>}
        <button onClick={handleSubmit} disabled={submitting || answeredCount < 40} className={`w-full font-bold py-4 rounded-full shadow-lg transition-all text-lg ${answeredCount < 40 ? 'bg-gray-300 text-gray-500 cursor-not-allowed shadow-none' : 'bg-[#2ECC71] hover:bg-green-500 text-white shadow-green-200'} disabled:opacity-60`}>
          Submit Answers
        </button>
      </div>
    </div>
  );
}

// ── RESULTS SCREEN ───────────────────────────────────────────────
function ResultsScreen({ onNavigate, sessionUUID }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!sessionUUID) { setError('No session found'); setLoading(false); return; }
    const loadReport = async () => {
      try {
        await ResultsAPI.fuse(sessionUUID);
        const data = await ResultsAPI.getReport(sessionUUID);
        setReport(data);
      } catch (err) {
        setError(err.message);
      } finally { setLoading(false); }
    };
    loadReport();
  }, [sessionUUID]);

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

  const fusionScore = report?.final_risk_score || 0;
  const videoScore = report?.video_score || 0;
  const questionnaireScore = report?.questionnaire_probability || 0;
  const videoProbs = report?.video_class_probabilities || {};
  const videoDetails = { headBanging: videoProbs.head_banging || 0, armFlapping: videoProbs.arm_flapping || 0, spinning: videoProbs.spinning || 0 };
  const catScores = report?.category_scores || {};
  const questionnaireDetails = { social: catScores.social_interaction || 0, communication: catScores.communication || 0, behavioral: catScores.behavior_patterns || 0, emotional: catScores.sensory_emotional || 0 };

  return (
    <div className="flex-1 bg-gray-50 flex flex-col h-full overflow-hidden relative">
      <div className="absolute top-4 left-4 z-20">
        <button onClick={() => onNavigate('home')} className="w-10 h-10 rounded-full bg-white shadow-md flex items-center justify-center text-gray-600 hover:bg-gray-50">
          <ArrowLeft className="w-5 h-5" />
        </button>
      </div>

      <PostAnalysisResult fusionScore={fusionScore} videoScore={videoScore} questionnaireScore={questionnaireScore} videoDetails={videoDetails} questionnaireDetails={questionnaireDetails} patientName="Your Child" />

      <div className="absolute bottom-0 left-0 right-0 bg-white p-4 pb-8 border-t border-gray-100 shadow-[0_-10px_20px_rgba(0,0,0,0.03)] z-20">
        <p className="text-center text-xs text-gray-500 mb-1 font-medium">Have questions about this report?</p>
        <button onClick={() => onNavigate('chatbot')} className="w-full bg-[#1A9E7A] text-white font-bold py-4 rounded-2xl shadow-lg shadow-teal-200 flex items-center justify-center gap-2 hover:bg-teal-700 transition-all text-lg">
          <Brain className="w-6 h-6" />
          <span>Consult AI Specialist Chatbot</span>
        </button>
      </div>
    </div>
  );
}

// ── CHATBOT SCREEN ───────────────────────────────────────────────
function ChatbotScreen({ onNavigate }) {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [conversationHistory, setConversationHistory] = useState([]);
  const endRef = useRef(null);

  useEffect(() => {
    const apiKey = import.meta.env.VITE_GROQ_API_KEY;
    if (!apiKey) {
      setMessages([{ id: 1, sender: 'ai', text: "Hello! I'm here to help you understand your child's screening results. How can I assist you today?", time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
      return;
    }

    setIsTyping(true);
    fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
      body: JSON.stringify({ model: 'llama-3.3-70b-versatile', messages: [{ role: 'user', content: "Generate a warm, empathetic opening message for a guardian who has just received their child's autism screening results. Keep it under 3 sentences." }], max_tokens: 150, temperature: 0.7 })
    }).then(res => res.json()).then(data => {
      setMessages([{ id: 1, sender: 'ai', text: data.choices[0].message.content, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
    }).catch(() => {
      setMessages([{ id: 1, sender: 'ai', text: "Hello! I'm here to help you understand your child's screening results.", time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
    }).finally(() => setIsTyping(false));
  }, []);

  const handleSend = async () => {
    if (!inputText.trim()) return;
    const userMessage = inputText.trim();
    setMessages(prev => [...prev, { id: Date.now(), sender: 'user', text: userMessage, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
    setInputText('');
    setIsTyping(true);

    const apiKey = import.meta.env.VITE_GROQ_API_KEY;
    if (!apiKey) {
      setMessages(prev => [...prev, { id: Date.now() + 1, sender: 'ai', text: "I'm having trouble connecting right now.", time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
      setIsTyping(false);
      return;
    }

    try {
      const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
        body: JSON.stringify({ model: 'llama-3.3-70b-versatile', messages: [...conversationHistory, { role: 'user', content: userMessage }], max_tokens: 500, temperature: 0.7 })
      });
      const data = await res.json();
      const botReply = data.choices[0].message.content;
      setConversationHistory(prev => [...prev, { role: 'user', content: userMessage }, { role: 'assistant', content: botReply }]);
      setMessages(prev => [...prev, { id: Date.now() + 1, sender: 'ai', text: botReply, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
    } catch {
      setMessages(prev => [...prev, { id: Date.now() + 1, sender: 'ai', text: "I'm having trouble connecting right now.", time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
    } finally { setIsTyping(false); }
  };

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, isTyping]);

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

      <div className="flex-1 overflow-y-auto p-5 md:p-8 space-y-4 md:space-y-6 md:max-w-4xl md:mx-auto md:w-full">
        {messages.map(msg => (
          <div key={msg.id} className={`flex flex-col max-w-[85%] ${msg.sender === 'user' ? 'ml-auto items-end' : 'mr-auto items-start'}`}>
            <div className={`px-4 py-3 rounded-2xl shadow-sm text-[15px] leading-relaxed ${msg.sender === 'user' ? 'bg-[#1A9E7A] text-white rounded-tr-sm' : 'bg-white border border-gray-100 text-gray-800 rounded-tl-sm'}`}>
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
          <input type="text" placeholder="Type your question..." className="flex-1 bg-transparent border-none focus:outline-none px-4 py-2 text-sm text-gray-800" value={inputText} onChange={(e) => setInputText(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSend()} />
          <button onClick={handleSend} disabled={!inputText.trim()} className="w-10 h-10 rounded-full bg-[#1A9E7A] text-white flex items-center justify-center disabled:bg-gray-300 transition-colors shadow-sm">
            <Send className="w-4 h-4 ml-0.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

// ── CLINICS SCREEN ───────────────────────────────────────────────
function ClinicsScreen({ onNavigate }) {
  return (
    <div className="flex-1 bg-gray-50 flex flex-col h-full overflow-hidden pb-16">
      {/* Header */}
      <div className="bg-[#1A9E7A] pt-5 px-5 pb-5 shadow-md z-10 relative shrink-0">
        <div className="flex items-center justify-between gap-4">
          <button onClick={() => onNavigate('home')} className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center text-white backdrop-blur-sm">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <h2 className="text-lg font-bold text-white">Nearby Specialists</h2>
          <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center text-white backdrop-blur-sm">
            <MapPin className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* Widget fills the remaining space */}
      <LocationSpecialistWidget
        geoapifyApiKey={import.meta.env.VITE_GEOAPIFY_API_KEY ?? ''}
      />
    </div>
  );
}

// ── CREATE ACCOUNT SCREEN ─────────────────────────────────────────
function CreateAccountScreen({ onNavigate }) {
  const { register } = useAuth();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleRegister = async () => {
    if (!email || !password) { setError('Please fill all fields'); return; }
    setLoading(true); setError('');
    try {
      await register(email, password, name || null);
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
          
          {error && <div className="w-full bg-red-50 border border-red-200 text-red-600 text-sm px-4 py-3 rounded-xl mb-4 font-medium">{error}</div>}

          <div className="w-full space-y-4 mb-8">
            <input type="text" placeholder="Full Name (optional)" value={name} onChange={e => setName(e.target.value)} className="w-full bg-white/80 border border-gray-200 rounded-2xl py-3.5 px-4 text-gray-700 focus:outline-none focus:ring-2 focus:ring-[#1A9E7A] focus:bg-white transition-all shadow-sm" />
            <input type="email" placeholder="Email address" value={email} onChange={e => setEmail(e.target.value)} className="w-full bg-white/80 border border-gray-200 rounded-2xl py-3.5 px-4 text-gray-700 focus:outline-none focus:ring-2 focus:ring-[#1A9E7A] focus:bg-white transition-all shadow-sm" />
            <input type="password" placeholder="Password (min 8 chars, 1 uppercase, 1 number)" value={password} onChange={e => setPassword(e.target.value)} className="w-full bg-white/80 border border-gray-200 rounded-2xl py-3.5 px-4 text-gray-700 focus:outline-none focus:ring-2 focus:ring-[#1A9E7A] focus:bg-white transition-all shadow-sm" />
          </div>
          
          <button onClick={handleRegister} disabled={loading} className="w-full bg-[#1A9E7A] hover:bg-teal-700 text-white font-bold py-4 rounded-full shadow-lg shadow-teal-200 hover:shadow-xl transition-all text-lg mb-4 disabled:opacity-50">
            {loading ? 'Creating account...' : 'Register'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── HISTORY SCREEN ───────────────────────────────────────────────
function HistoryScreen({ onNavigate }) {
  const { isAuthenticated } = useAuth();
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadHistory() {
      if (!isAuthenticated) { setLoading(false); return; }
      try {
        const data = await AuthAPI.getSessionHistory();
        setSessions(data.sessions || []);
      } catch (err) {
        console.error('Failed to load history:', err);
      } finally { setLoading(false); }
    }
    loadHistory();
  }, [isAuthenticated]);

  const formatDate = (dateStr) => new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  const getRiskColor = (level) => {
    if (level === 'high') return 'bg-red-50 text-red-600 border-red-100';
    if (level === 'medium') return 'bg-orange-50 text-orange-600 border-orange-100';
    return 'bg-green-50 text-green-600 border-green-100';
  };

  if (!isAuthenticated) {
    return (
      <div className="flex-1 bg-gray-50 flex flex-col items-center justify-center p-6 pb-20">
        <p className="text-gray-500 mb-4">Sign in to view your assessment history</p>
        <button onClick={() => onNavigate('login')} className="bg-[#1A9E7A] text-white px-6 py-3 rounded-full font-bold">Sign In</button>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex-1 bg-gray-50 flex items-center justify-center pb-20">
        <div className="w-12 h-12 border-4 border-teal-200 border-t-[#1A9E7A] rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-gray-50 flex flex-col h-full overflow-hidden pb-20">
      <div className="px-6 py-5 bg-white border-b border-gray-100 shadow-sm z-10">
        <h2 className="text-xl font-bold text-gray-900">Assessment History</h2>
      </div>
      
      <div className="flex-1 overflow-y-auto p-6">
        {sessions.length === 0 ? (
          <div className="text-center text-gray-500 py-12">
            <p>No completed assessments yet</p>
            <button onClick={() => onNavigate('home')} className="mt-4 text-[#1A9E7A] font-bold">Start your first assessment</button>
          </div>
        ) : (
          <div className="space-y-4 md:max-w-4xl md:mx-auto md:w-full">
            {sessions.map((session, i) => (
              <div key={session.session_uuid || i} className="bg-white rounded-2xl p-5 shadow-sm border border-gray-100 flex justify-between items-center">
                <div>
                  <p className="font-bold text-gray-900 text-lg mb-1.5">{formatDate(session.created_at)}</p>
                  <span className={`text-xs font-bold px-2 py-1 rounded-md border ${getRiskColor(session.risk_level)}`}>
                    {session.risk_level ? `${session.risk_level.charAt(0).toUpperCase() + session.risk_level.slice(1)} Risk` : 'Pending'}
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-2xl font-black text-gray-800">{session.final_risk_score ? `${Math.round(session.final_risk_score * 100)}%` : '—'}</span>
                  <p className="text-xs text-gray-400 font-bold uppercase tracking-widest mt-0.5">Risk</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── PROFILE SCREEN ───────────────────────────────────────────────
function ProfileScreen({ onNavigate, onLogout }) {
  const { user, updateDisplayName, isGuest } = useAuth();
  const [isEditing, setIsEditing] = useState(false);
  const [nameInput, setNameInput] = useState(user?.display_name || '');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const displayName = user?.display_name || 'Guest';

  const handleSaveName = async () => {
    if (!nameInput.trim()) { setError('Name cannot be empty'); return; }
    setSaving(true); setError('');
    try {
      await updateDisplayName(nameInput.trim());
      setIsEditing(false);
    } catch (err) {
      setError(err.message);
    } finally { setSaving(false); }
  };

  return (
    <div className="flex-1 bg-gray-50 flex flex-col h-full overflow-hidden pb-20">
      <div className="px-6 py-5 bg-white border-b border-gray-100 shadow-sm z-10">
        <h2 className="text-xl font-bold text-gray-900">Profile</h2>
      </div>
      
      <div className="flex-1 overflow-y-auto p-6 md:p-10 md:max-w-3xl md:mx-auto md:w-full">
        <div className="bg-white rounded-3xl p-6 md:p-10 shadow-sm border border-gray-100 flex flex-col items-center mb-6 relative overflow-hidden">
          <div className="absolute top-0 w-full h-16 md:h-24 bg-teal-50"></div>
          <div className="w-24 h-24 bg-white border-4 border-white shadow-sm text-teal-700 rounded-full flex items-center justify-center mb-4 z-10">
            <User className="w-10 h-10 text-teal-600" />
          </div>
          
          {isEditing ? (
            <div className="flex flex-col items-center gap-3 z-10 w-full max-w-sm">
              <input type="text" value={nameInput} onChange={(e) => setNameInput(e.target.value)} className="w-full bg-gray-50 border border-gray-200 rounded-xl py-3 px-4 text-center text-lg font-semibold focus:outline-none focus:ring-2 focus:ring-[#1A9E7A]" placeholder="Enter your name" autoFocus />
              {error && <p className="text-red-500 text-sm">{error}</p>}
              <div className="flex gap-2">
                <button onClick={() => { setIsEditing(false); setError(''); }} className="px-4 py-2 rounded-lg bg-gray-100 text-gray-600 font-medium">Cancel</button>
                <button onClick={handleSaveName} disabled={saving} className="px-4 py-2 rounded-lg bg-[#1A9E7A] text-white font-medium flex items-center gap-1">
                  {saving ? 'Saving...' : <><Check className="w-4 h-4" /> Save</>}
                </button>
              </div>
            </div>
          ) : (
            <h3 className="text-xl font-bold text-gray-900 z-10">{displayName}</h3>
          )}
        </div>

        <div className="space-y-3">
          {!isGuest && (
            <button onClick={() => { setNameInput(displayName); setIsEditing(true); }} className="w-full bg-white p-4 rounded-2xl border border-gray-100 flex items-center justify-between shadow-sm text-gray-700 font-semibold hover:border-teal-200 hover:shadow-md hover:bg-teal-50">
              Change Name
              <ChevronRight className="w-5 h-5 text-gray-300" />
            </button>
          )}

          <button onClick={onLogout} className="w-full bg-white p-4 rounded-2xl border border-gray-100 flex items-center justify-center shadow-sm text-red-500 font-bold hover:bg-red-50 mt-4">
            Log Out
          </button>
        </div>

        {isGuest && (
          <p className="text-center text-gray-500 text-sm mt-6">Create an account to save your assessment history</p>
        )}
      </div>
    </div>
  );
}
