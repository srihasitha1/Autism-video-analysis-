// src/services/api.js
const API_BASE = '/api/v1';

// Token management
export const TokenManager = {
  get: () => localStorage.getItem('auth_token'),
  set: (token) => localStorage.setItem('auth_token', token),
  clear: () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('session_uuid');
  },
  isValid: () => !!localStorage.getItem('auth_token'),
};

// Base fetch with auth header
export async function apiFetch(endpoint, options = {}) {
  const token = TokenManager.get();
  const headers = {
    'Content-Type': 'application/json',
    ...(token && { Authorization: `Bearer ${token}` }),
    ...options.headers,
  };

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (response.status === 401) {
    TokenManager.clear();
    window.dispatchEvent(new CustomEvent('auth-expired'));
  }

  return response;
}

// Auth API
export const AuthAPI = {
  async login(email, password) {
    const res = await apiFetch('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Login failed');
    }
    const data = await res.json();
    TokenManager.set(data.access_token);
    return data;
  },

  async register(email, password, displayName = null) {
    const res = await apiFetch('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Registration failed');
    }
    const data = await res.json();
    TokenManager.set(data.access_token);
    
    // Set display name if provided
    if (displayName) {
      try {
        await this.updateProfile(displayName);
      } catch (err) {
        console.warn('Could not set display name:', err);
      }
    }
    return data;
  },

  async getProfile() {
    const res = await apiFetch('/auth/me');
    if (!res.ok) return { is_guest: true, display_name: 'Guest' };
    return res.json();
  },

  async updateProfile(displayName) {
    const res = await apiFetch('/auth/me', {
      method: 'PUT',
      body: JSON.stringify({ display_name: displayName }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Update failed');
    }
    return res.json();
  },

  async getSessionHistory() {
    const res = await apiFetch('/auth/sessions');
    if (!res.ok) return { sessions: [] };
    return res.json();
  },

  async createGuestSession() {
    const res = await apiFetch('/auth/guest', { method: 'POST' });
    if (!res.ok) throw new Error('Failed to create session');
    return res.json();
  },

  logout() {
    TokenManager.clear();
  },
};

// Video API
export const VideoAPI = {
  async upload(sessionUuid, file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_uuid', sessionUuid);
    
    const token = TokenManager.get();
    const res = await fetch(`${API_BASE}/analyze/video/upload`, {
      method: 'POST',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: formData,
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Upload failed');
    }
    return res.json();
  },

  async startAnalysis(sessionUuid) {
    const res = await apiFetch('/analyze/video/start', {
      method: 'POST',
      body: JSON.stringify({ session_uuid: sessionUuid }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Analysis start failed');
    }
    return res.json();
  },

  async getStatus(sessionUuid) {
    const res = await apiFetch(`/analyze/video/status/${sessionUuid}`);
    if (!res.ok) throw new Error('Status check failed');
    return res.json();
  },
};

// Questionnaire API
export const QuestionnaireAPI = {
  async submit(sessionUuid, responses, childAgeMonths = 36, childGender = 'unspecified') {
    const res = await apiFetch('/analyze/questionnaire', {
      method: 'POST',
      body: JSON.stringify({
        session_uuid: sessionUuid,
        responses,
        child_age_months: childAgeMonths,
        child_gender: childGender,
      }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Submission failed');
    }
    return res.json();
  },
};

// Fusion/Results API
export const ResultsAPI = {
  async fuse(sessionUuid) {
    const res = await apiFetch('/analyze/fuse', {
      method: 'POST',
      body: JSON.stringify({ session_uuid: sessionUuid }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Fusion failed');
    }
    return res.json();
  },

  async getReport(sessionUuid) {
    const res = await apiFetch(`/analyze/report/${sessionUuid}`);
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Report fetch failed');
    }
    return res.json();
  },
};
