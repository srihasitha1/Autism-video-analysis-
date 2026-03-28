import React, { useState, useRef } from 'react';
import { MapPin, Search, Loader, AlertTriangle, Phone, Globe, Clock, Navigation } from 'lucide-react';

// ── Fallback demo data (shown when API key is empty) ───────────────────────────
const DEMO_CLINICS = [
  {
    name: 'Child Development & Autism Center',
    address: 'North Wing, City Hospital',
    distanceMeters: 1200,
    distanceLabel: '1.2 km away',
    phone: null,
    website: null,
    hours: 'Mon–Fri 9am–5pm',
    lng: null,
    lat: null,
  },
  {
    name: 'Pediatric Neurology Associates',
    address: 'Health Plaza, Suite 204',
    distanceMeters: 3500,
    distanceLabel: '3.5 km away',
    phone: null,
    website: null,
    hours: null,
    lng: null,
    lat: null,
  },
  {
    name: 'Early Childhood Support Institute',
    address: 'Community Care Building',
    distanceMeters: 5100,
    distanceLabel: '5.1 km away',
    phone: null,
    website: null,
    hours: null,
    lng: null,
    lat: null,
  },
];

// ── Geoapify Places API ────────────────────────────────────────────────────────
async function fetchNearbyClinics(lat, lng, apiKey) {
  const url = new URL('https://api.geoapify.com/v2/places');
  url.searchParams.set('categories', 'healthcare.hospital,healthcare.clinic_or_praxis');
  url.searchParams.set('filter', `circle:${lng},${lat},15000`);
  url.searchParams.set('bias', `proximity:${lng},${lat}`);
  url.searchParams.set('limit', '20');
  url.searchParams.set('apiKey', apiKey);

  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`Places API error: ${res.status}`);
  const json = await res.json();

  const features = json.features || [];
  return features
    .filter((f) => f.properties?.name)
    .map((f) => ({
      name: f.properties.name,
      address: [f.properties.address_line1, f.properties.address_line2]
        .filter(Boolean)
        .join(', '),
      distanceMeters: f.properties.distance ?? 0,
      distanceLabel: `${((f.properties.distance ?? 0) / 1000).toFixed(1)} km away`,
      phone: f.properties.datasource?.raw?.phone ?? null,
      website: f.properties.datasource?.raw?.website ?? null,
      hours: f.properties.datasource?.raw?.opening_hours ?? null,
      lng: f.geometry.coordinates[0],
      lat: f.geometry.coordinates[1],
    }))
    .sort((a, b) => a.distanceMeters - b.distanceMeters)
    .slice(0, 10);
}

// ── Geoapify Geocoding API ─────────────────────────────────────────────────────
async function geocodeLocation(text, apiKey) {
  const url = new URL('https://api.geoapify.com/v1/geocode/search');
  url.searchParams.set('text', text);
  url.searchParams.set('limit', '1');
  url.searchParams.set('apiKey', apiKey);

  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`Geocoding API error: ${res.status}`);
  const json = await res.json();

  const features = json.features || [];
  if (features.length === 0) return null;

  const [lng, lat] = features[0].geometry.coordinates;
  return { lat, lng, label: features[0].properties?.formatted ?? text };
}

// ── Geoapify Static Maps URL ───────────────────────────────────────────────────
function buildStaticMapUrl(userLat, userLng, clinics, apiKey) {
  const params = new URLSearchParams({
    style: 'osm-bright',
    width: '600',
    height: '300',
    center: `lonlat:${userLng},${userLat}`,
    zoom: '13',
    apiKey,
  });

  // user marker (blue)
  params.append('marker', `lonlat:${userLng},${userLat};color:%232563EB;size:medium`);

  // clinic markers (red) — up to 5
  clinics.slice(0, 5).forEach((c) => {
    if (c.lng !== null && c.lat !== null) {
      params.append('marker', `lonlat:${c.lng},${c.lat};color:%23EF4444;size:small`);
    }
  });

  return `https://maps.geoapify.com/v1/staticmap?${params.toString()}`;
}

// ── Main component ─────────────────────────────────────────────────────────────
export default function LocationSpecialistWidget({ geoapifyApiKey = '' }) {
  const [phase, setPhase] = useState('idle'); // idle | requesting | loading | success | error | empty
  const [clinics, setClinics] = useState([]);
  const [userCoords, setUserCoords] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [mapLoaded, setMapLoaded] = useState(false);
  const [mapError, setMapError] = useState(false);

  // Manual search
  const [searchText, setSearchText] = useState('');
  const [searchError, setSearchError] = useState('');
  const [searching, setSearching] = useState(false);

  const isDemo = !geoapifyApiKey;

  // ── Run clinic fetch after coords are known ─────────────────────────────────
  async function loadClinics(lat, lng) {
    setPhase('loading');
    if (isDemo) {
      // Short artificial delay for UX continuity
      await new Promise((r) => setTimeout(r, 800));
      setClinics(DEMO_CLINICS);
      setUserCoords({ lat, lng });
      setPhase('success');
      return;
    }

    try {
      const results = await fetchNearbyClinics(lat, lng, geoapifyApiKey);
      setUserCoords({ lat, lng });
      if (results.length === 0) {
        setPhase('empty');
      } else {
        setClinics(results);
        setPhase('success');
      }
    } catch (err) {
      setErrorMsg(err.message || 'Failed to fetch nearby clinics.');
      setPhase('error');
    }
  }

  // ── Browser geolocation ─────────────────────────────────────────────────────
  function handleUseMyLocation() {
    setPhase('requesting');
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        loadClinics(pos.coords.latitude, pos.coords.longitude);
      },
      (err) => {
        setErrorMsg(
          err.code === 1
            ? 'Location access denied. Please allow location access or search manually.'
            : 'Unable to determine your location. Please try searching manually.'
        );
        setPhase('error');
      },
      { timeout: 15000 }
    );
  }

  // ── Manual search / geocoding ───────────────────────────────────────────────
  async function handleSearch() {
    if (!searchText.trim()) return;
    setSearchError('');
    setSearching(true);

    if (isDemo) {
      setSearching(false);
      await loadClinics(51.5074, -0.1278); // demo coords (London)
      return;
    }

    try {
      const result = await geocodeLocation(searchText.trim(), geoapifyApiKey);
      if (!result) {
        setSearchError('Location not found. Please try a different city or postcode.');
        setSearching(false);
        return;
      }
      setSearching(false);
      loadClinics(result.lat, result.lng);
    } catch (err) {
      setSearchError('Geocoding failed. Please try again.');
      setSearching(false);
    }
  }

  // ── Spinner helper ───────────────────────────────────────────────────────────
  const Spinner = ({ label }) => (
    <div className="flex-1 flex flex-col items-center justify-center gap-4 p-8">
      <div className="w-14 h-14 border-4 border-teal-100 border-t-[#1A9E7A] rounded-full animate-spin" />
      <p className="text-gray-500 font-medium text-center">{label}</p>
    </div>
  );

  // ── IDLE STATE ───────────────────────────────────────────────────────────────
  if (phase === 'idle') {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 gap-6 bg-gray-50 min-h-0 overflow-y-auto pb-10">
        {/* Icon */}
        <div className="w-20 h-20 rounded-full bg-teal-50 border-2 border-teal-100 flex items-center justify-center shadow-inner">
          <MapPin className="w-9 h-9 text-[#1A9E7A]" />
        </div>

        {/* Heading */}
        <div className="text-center max-w-xs">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Find Nearby Clinics</h2>
          <p className="text-gray-500 text-sm leading-relaxed">
            We'll find autism specialists and child development centers near you. Your location
            stays private and is never stored.
          </p>
        </div>

        {/* Manual search */}
        <div className="w-full max-w-sm space-y-2">
          <label className="block text-sm font-semibold text-gray-600 mb-1">
            Or enter your city or postcode
          </label>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
              <input
                id="location-search-input"
                type="text"
                placeholder="e.g. London, EC1A 1BB"
                value={searchText}
                onChange={(e) => { setSearchText(e.target.value); setSearchError(''); }}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                className="w-full border border-gray-200 rounded-xl py-2.5 pl-9 pr-4 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-[#1A9E7A] bg-white shadow-sm"
              />
            </div>
            <button
              id="location-search-btn"
              onClick={handleSearch}
              disabled={searching || !searchText.trim()}
              className="px-4 py-2.5 bg-[#1A9E7A] text-white font-bold rounded-xl text-sm disabled:opacity-50 hover:bg-teal-700 transition-colors shadow-sm"
            >
              {searching ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : 'Search'}
            </button>
          </div>
          {searchError && (
            <p className="text-red-500 text-xs font-medium mt-1">{searchError}</p>
          )}
        </div>

        {/* Divider */}
        <div className="flex items-center gap-3 w-full max-w-sm">
          <div className="flex-1 h-px bg-gray-200" />
          <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">or</span>
          <div className="flex-1 h-px bg-gray-200" />
        </div>

        {/* GPS button */}
        <button
          id="use-my-location-btn"
          onClick={handleUseMyLocation}
          className="w-full max-w-sm flex items-center justify-center gap-3 bg-[#1A9E7A] hover:bg-teal-700 text-white font-bold py-4 rounded-2xl shadow-lg shadow-teal-100 transition-all text-base"
        >
          <Navigation className="w-5 h-5" />
          Use My Location
        </button>
      </div>
    );
  }

  // ── REQUESTING / LOADING ─────────────────────────────────────────────────────
  if (phase === 'requesting') return <Spinner label="Finding your location…" />;
  if (phase === 'loading') return <Spinner label="Searching for nearby clinics…" />;

  // ── ERROR STATE ──────────────────────────────────────────────────────────────
  if (phase === 'error') {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 gap-4 text-center">
        <div className="w-16 h-16 rounded-full bg-red-50 flex items-center justify-center">
          <AlertTriangle className="w-8 h-8 text-red-400" />
        </div>
        <p className="text-red-600 font-semibold max-w-xs leading-relaxed">{errorMsg}</p>
        <button
          onClick={() => { setErrorMsg(''); setPhase('idle'); }}
          className="mt-2 bg-[#1A9E7A] text-white font-bold px-8 py-3 rounded-xl hover:bg-teal-700 transition-colors"
        >
          Try Again
        </button>
      </div>
    );
  }

  // ── EMPTY STATE ──────────────────────────────────────────────────────────────
  if (phase === 'empty') {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 gap-4 text-center">
        <div className="w-16 h-16 rounded-full bg-amber-50 flex items-center justify-center">
          <MapPin className="w-8 h-8 text-amber-400" />
        </div>
        <p className="text-gray-600 font-semibold max-w-xs leading-relaxed">
          No clinics found within 15 km. Try searching a nearby city.
        </p>
        <button
          onClick={() => setPhase('idle')}
          className="mt-2 bg-[#1A9E7A] text-white font-bold px-8 py-3 rounded-xl hover:bg-teal-700 transition-colors"
        >
          Search Again
        </button>
      </div>
    );
  }

  // ── SUCCESS STATE ────────────────────────────────────────────────────────────
  const mapUrl = !isDemo && userCoords
    ? buildStaticMapUrl(userCoords.lat, userCoords.lng, clinics, geoapifyApiKey)
    : null;

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {/* Demo notice bar */}
      {isDemo && (
        <div className="bg-amber-50 border-b border-amber-200 px-4 py-2.5 text-amber-700 text-xs font-semibold flex items-center gap-2 shrink-0">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          Showing demo results. Add a Geoapify API key for live clinic data.
        </div>
      )}

      {/* Static map section */}
      {mapUrl && (
        <div className="shrink-0 relative bg-gray-100" style={{ height: '200px' }}>
          {!mapLoaded && !mapError && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-100 animate-pulse">
              <Loader className="w-6 h-6 text-gray-400 animate-spin" />
            </div>
          )}
          {mapError ? (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
              <p className="text-gray-400 text-sm font-medium">Map preview unavailable.</p>
            </div>
          ) : (
            <img
              src={mapUrl}
              alt="Map of nearby clinics"
              className={`w-full h-full object-cover transition-opacity duration-300 ${mapLoaded ? 'opacity-100' : 'opacity-0'}`}
              style={{ width: '100%', height: '200px' }}
              onLoad={() => setMapLoaded(true)}
              onError={() => { setMapError(true); setMapLoaded(false); }}
            />
          )}
        </div>
      )}

      {/* Header */}
      <div className="px-5 py-3 border-b border-gray-100 bg-white shrink-0 flex items-center justify-between">
        <h3 className="font-bold text-gray-900 text-base">
          Nearest Clinics
          <span className="ml-2 text-sm font-medium text-gray-400">({clinics.length} found)</span>
        </h3>
        <button
          onClick={() => { setClinics([]); setUserCoords(null); setMapLoaded(false); setMapError(false); setPhase('idle'); }}
          className="text-[#1A9E7A] text-xs font-bold hover:underline"
        >
          Search Again
        </button>
      </div>

      {/* Scrollable clinic list */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {clinics.map((clinic, i) => (
          <ClinicCard key={i} clinic={clinic} />
        ))}
      </div>
    </div>
  );
}

// ── Clinic card sub-component ──────────────────────────────────────────────────
function ClinicCard({ clinic }) {
  return (
    <div className="bg-white rounded-2xl p-4 shadow-sm border border-gray-100 space-y-2">
      {/* Name + distance */}
      <div className="flex items-start justify-between gap-2">
        <h4 className="font-bold text-gray-900 text-[15px] leading-snug">{clinic.name}</h4>
        <span className="shrink-0 text-xs font-semibold text-[#1A9E7A] bg-teal-50 px-2 py-1 rounded-lg whitespace-nowrap">
          {clinic.distanceLabel}
        </span>
      </div>

      {/* Address */}
      {clinic.address && (
        <p className="text-gray-500 text-sm flex items-start gap-1.5">
          <MapPin className="w-3.5 h-3.5 text-gray-400 mt-0.5 shrink-0" />
          {clinic.address}
        </p>
      )}

      {/* Optional details */}
      <div className="space-y-1">
        {clinic.hours && (
          <p className="text-gray-500 text-xs flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-gray-400 shrink-0" />
            {clinic.hours}
          </p>
        )}
        {clinic.phone && (
          <a
            href={`tel:${clinic.phone}`}
            className="text-[#1A9E7A] text-xs font-medium flex items-center gap-1.5 hover:underline"
          >
            <Phone className="w-3.5 h-3.5 shrink-0" />
            {clinic.phone}
          </a>
        )}
        {clinic.website && (
          <a
            href={clinic.website}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-500 text-xs font-medium flex items-center gap-1.5 hover:underline truncate"
          >
            <Globe className="w-3.5 h-3.5 shrink-0" />
            {clinic.website.replace(/^https?:\/\//, '')}
          </a>
        )}
      </div>
    </div>
  );
}
