import React, { useState, useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import { RefreshCw, AlertTriangle, Navigation, Sprout, Search } from "lucide-react";
import "leaflet/dist/leaflet.css";

// ----------------- PUT YOUR KEY HERE -----------------
const LOCATIONIQ_API_KEY = "pk.08477d598dd0aa98ede039a9ef45df6a";
// -----------------------------------------------------

// leaflet default icon fix
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

// Hardcoded prominent shops in Chennai for reliable initial results
// Coordinates updated based on provided addresses for better accuracy
const HARDCODED_CHENNAI_SHOPS = [
    { name: "Susi Seeds and Hasini seeds (High Priority)", lat: 13.0487, lon: 80.2461, distanceMeters: 0, hardcoded: true, address: { road: "Natesan Street", town: "T. Nagar, Chennai" } },
    { name: "Sakthi Seeds (High Priority)", lat: 13.0488, lon: 80.2461, distanceMeters: 0, hardcoded: true, address: { road: "Natesan Street", town: "T. Nagar, Chennai" } },
    { name: "Dakshu Seeds Pvt. Ltd. (High Priority)", lat: 13.0805, lon: 80.2118, distanceMeters: 0, hardcoded: true, address: { road: "Anna Nagar 3rd Main Rd", town: "Anna Nagar, Chennai" } },
    { name: "Tamil Nadu Seed Corporation (TANSCO)", lat: 13.067439, lon: 80.237617, distanceMeters: 0, hardcoded: true, address: { town: "Chennai, TN" } },
    { name: "Madras Fertilizers Ltd.", lat: 13.08268, lon: 80.27072, distanceMeters: 0, hardcoded: true, address: { town: "Manali, Chennai" } },
];

// keywords for filtering
const ALLOWED_KEYWORDS = [
  "seed",
  "fertilizer",
  "fertiliser",
  "pesticide",
  "agro",
  "agri",
  "krishi",
  "krushi",
  "farm",
  "horticulture",
  "nursery",
  "garden",
  "agrovet",
  "agriculture",
];

function toNum(v) {
  if (typeof v === "number") return v;
  if (!v) return 0;
  return Number(v);
}

// Haversine distance (meters)
function haversine(lat1, lon1, lat2, lon2) {
  const toRad = (v) => (v * Math.PI) / 180;
  const R = 6371000;
  const φ1 = toRad(lat1), φ2 = toRad(lat2);
  const Δφ = toRad(lat2 - lat1), Δλ = toRad(lon2 - lon1);
  const a = Math.sin(Δφ/2)**2 + Math.cos(φ1)*Math.cos(φ2)*Math.sin(Δλ/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.atan2(Math.sqrt(a), Math.sqrt(1-a)));
}

// basic agri filter
function isAgriShop(item) {
  const name = (item.name || item.display_name || "").toLowerCase();
  const addr = JSON.stringify(item.address || "").toLowerCase();
  const tags = JSON.stringify(item.extratags || item.tags || {}).toLowerCase();

  return ALLOWED_KEYWORDS.some(k => name.includes(k) || addr.includes(k) || tags.includes(k));
}

function Recenter({ lat, lon, zoom }) {
  const map = useMap();
  useEffect(() => {
    if (lat && lon) map.setView([lat, lon], zoom);
  }, [lat, lon, zoom, map]);
  return null;
}

export default function SeedShopFinder() {
  const DEFAULT_CHENNAI_LOC = { lat: 13.0487, lon: 80.2461 }; // Central T. Nagar area
  const [location, setLocation] = useState(DEFAULT_CHENNAI_LOC);
  const [shops, setShops] = useState(HARDCODED_CHENNAI_SHOPS);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // Set initial search query to be relevant to the hardcoded location
  const [searchQuery, setSearchQuery] = useState("seed and fertilizer shops in Chennai");
  const [currentZoom, setCurrentZoom] = useState(13);


  // Wrapper for LocationIQ Text Search (used by manual search and geolocation)
  const searchPipeline = async (query, centerLat, centerLon) => {
    setLoading(true);
    setError(""); // Clear previous errors
    
    // Fallback query if input is empty
    const finalQuery = query.trim() || "agro shop";
    // Set a search area box 
    const center = centerLat && centerLon ? `&viewbox=${centerLon-0.2},${centerLat-0.2},${centerLon+0.2},${centerLat+0.2}&bounded=1` : "";

    try {
        // Use the LocationIQ Search API for general text search
        const q = encodeURIComponent(finalQuery);
        const url = `https://us1.locationiq.com/v1/search?key=${LOCATIONIQ_API_KEY}&q=${q}&format=json&limit=30${center}`;
        
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        const results = await response.json();
        
        // --- DATA PROCESSING AND FILTERING ---
        let unique = [];

        if (Array.isArray(results) && results.length > 0) {
            // Normalize lat/lon and filter agri shops
            const normalized = results
                .map(it => {
                    return {
                        ...it,
                        lat: toNum(it.lat),
                        lon: toNum(it.lon),
                        name: it.name || it.display_name || ""
                    };
                })
                .filter(it => it.lat && it.lon)
                .filter(isAgriShop);

            // Dedupe by rounded coords
            const map = new Map();
            normalized.forEach(it => {
                const k = `${it.lat.toFixed(5)}|${it.lon.toFixed(5)}`;
                if (!map.has(k)) map.set(k, it);
            });
            unique = Array.from(map.values());
        }

        // Always include hardcoded shops
        const finalShops = [...HARDCODED_CHENNAI_SHOPS];
        
        // Add unique search results, avoiding duplicates near hardcoded points
        unique.forEach(searchShop => {
            const isDuplicate = finalShops.some(hardcodedShop => 
                haversine(hardcodedShop.lat, hardcodedShop.lon, searchShop.lat, searchShop.lon) < 50 // 50 meters tolerance
            );
            if (!isDuplicate) {
                finalShops.push(searchShop);
            }
        });
        
        // Calculate distance only if user's location is known
        const centerPoint = centerLat && centerLon ? { lat: centerLat, lon: centerLon } : null;
        
        const withDist = finalShops.map(it => ({ 
            ...it, 
            distanceMeters: centerPoint ? haversine(centerPoint.lat, centerPoint.lon, it.lat, it.lon) : 0
        })).sort((a,b) => a.distanceMeters - b.distanceMeters);

        setShops(withDist.slice(0, 60)); // Limit to 60 results
        
        // Find center of results for map view
        if (withDist.length > 0) {
            // Set the map center to either the user's location or the default location for the search
            const initialLocation = centerPoint || { lat: withDist[0].lat, lon: withDist[0].lon };
            setLocation(initialLocation);
            setCurrentZoom(13); // Zoom in when results are good
        } else {
             // If no results, still clear error and just show hardcoded
             setError(""); 
             setShops(HARDCODED_CHENNAI_SHOPS);
             setLocation(DEFAULT_CHENNAI_LOC);
             setCurrentZoom(11); // Zoom out to show Chennai area
        }
        
    } catch (err) {
      console.error(err);
      // IMPORTANT: If API fails, suppress the error message but fall back to hardcoded shops
      setError(""); 
      setShops(HARDCODED_CHENNAI_SHOPS); // Fallback
      setLocation(DEFAULT_CHENNAI_LOC);
      setCurrentZoom(11);
    } finally {
      setLoading(false);
    }
  };
  
  // Gets user's GPS and triggers search (used by 'My Location' button)
  const getAndSearchNearMe = () => {
    setError("");
    setLoading(true);

    if (!navigator.geolocation) {
      setError("Geolocation not supported. Please use the search bar.");
      setLoading(false);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      pos => {
        const loc = { lat: pos.coords.latitude, lon: pos.coords.longitude };
        setLocation(loc);
        setSearchQuery("Agri shop near me");
        // Start search pipeline centered on user's GPS
        searchPipeline("Agri shop near me", loc.lat, loc.lon);
      },
      err => {
        // GPS error - set a temporary user-friendly message, but let the search pipeline handle the ultimate fallback
        setError("Could not get GPS. Showing default shops in Chennai.");
        setLoading(false);
        // Fallback to initial search on error
        searchPipeline(searchQuery, DEFAULT_CHENNAI_LOC.lat, DEFAULT_CHENNAI_LOC.lon); 
      },
      { enableHighAccuracy: true, timeout: 15000 }
    );
  };
  
  const handleSearch = (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    
    // Use the current map center (which is usually the last successful location or default Chennai)
    searchPipeline(searchQuery, location.lat, location.lon);
  }

  // Initial load effect (loads hardcoded shops and performs initial search)
  useEffect(() => {
    searchPipeline(searchQuery, location.lat, location.lon);
  }, []);

  return (
    <div className="finder-content"> 
        <header className="content-header">
            <div>
                <h1><Sprout /> Seed Shop Finder</h1>
                <p className="subtitle">Search for seed, fertilizer, and agro input shops</p>
            </div>
            {/* The general refresh/location button is moved below the search bar */}
        </header>
        
        {/* NEW SEARCH BAR */}
        <form onSubmit={handleSearch} className="input-group" style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
            <div style={{ flexGrow: 1, display: 'flex', alignItems: 'center', background: 'white', border: '1px solid #e2e8f0', borderRadius: 12, padding: '0 12px' }}>
                <Search size={18} color="#94a3b8" />
                <input
                    type="text"
                    placeholder="Search for 'seed shop in Coimbatore' or 'fertilizer store'"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    style={{ border: 'none', outline: 'none', padding: '12px 0', flexGrow: 1, marginLeft: 10, fontSize: '1rem' }}
                    disabled={loading}
                />
            </div>
            <button type="submit" className="primary-btn" disabled={loading || !searchQuery.trim()} style={{ padding: '0 20px', fontSize: '1rem' }}>
                Go
            </button>
            <button type="button" className="refresh-btn" onClick={getAndSearchNearMe} disabled={loading} title="Find Shops Near My Current GPS Location">
                {loading ? <RefreshCw size={18} className="animate-spin" /> : <Navigation size={18} />} GPS
            </button>
        </form>

        {error && (
            <div className="card danger-theme" style={{ marginBottom: 16 }}>
                <div className="action-header">
                    <AlertTriangle size={20} />
                    <p className="card-label">Problem</p>
                </div>
                <p>{error}</p>
            </div>
        )}

        <div className="card" style={{ padding: 0, height: "55vh", overflow: "hidden", marginBottom: 16 }}> 
            <MapContainer
                center={location}
                zoom={currentZoom}
                style={{ height: "100%", width: "100%" }}
            >
                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

                {location && (
                    <>
                        <Marker position={[location.lat, location.lon]}>
                            <Popup>Search Center or Your Location</Popup>
                        </Marker>
                        <Recenter lat={location.lat} lon={location.lon} zoom={currentZoom} />
                    </>
                )}

                {shops.map((s, i) => {
                    const shopName = s.name || s.display_name || "Agri Shop";
                    // Encode the shop name and coordinates for the Google Maps query
                    const mapQuery = encodeURIComponent(`${shopName}, ${s.lat},${s.lon}`);

                    return (
                        // Use a slightly different opacity/color for hardcoded shops
                        <Marker key={i} position={[s.lat, s.lon]} opacity={s.hardcoded ? 0.8 : 1}>
                            <Popup>
                                <div style={{ minWidth: 180 }}>
                                    <strong>{shopName}</strong>
                                    {s.hardcoded && <span style={{fontSize: 12, color: '#2563eb', marginLeft: 8}}>(High Priority)</span>}
                                    <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
                                        {s.address?.road ? s.address.road + ", " : ""}
                                        {s.address?.town || s.address?.village || s.address?.city || s.address?.suburb || ""}
                                    </div>
                                    <div style={{ marginTop: 8 }}>
                                        <a 
                                            // UPDATED: Using a query string that includes the shop name and coordinates.
                                            // This is generally more reliable for pinpointing a location and showing info.
                                            href={`https://www.google.com/maps/search/?api=1&query=${mapQuery}`} 
                                            target="_blank" 
                                            rel="noreferrer"
                                        >
                                            <Navigation size={14} /> Open in Maps
                                        </a>
                                        <div style={{ fontSize: 12, marginTop: 6, color: "var(--text-secondary)" }}>
                                            {s.distanceMeters ? `${(s.distanceMeters/1000).toFixed(2)} km` : (s.hardcoded ? "Pre-loaded Shop" : "")}
                                        </div>
                                    </div>
                                </div>
                            </Popup>
                        </Marker>
                    );
                })}
            </MapContainer>
        </div>

        <div style={{ marginTop: 12 }}>
            {loading && <p>Searching for shops...</p>}

            {!loading && shops.length > 0 && (
                <>
                    <h3 className="section-title">Nearby results ({shops.length})</h3>
                    <div style={{ display: "grid", gap: 10 }}>
                        {shops.map((s, i) => {
                            const shopName = s.name || s.display_name || "Agri Shop";
                            const mapQuery = encodeURIComponent(`${shopName}, ${s.lat},${s.lon}`);

                            return (
                                <div key={i} className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                    <div>
                                        <div style={{ fontWeight: 800 }}>{shopName} {s.hardcoded && <span style={{fontSize: 12, color: '#2563eb', fontWeight: 500}}>(Priority)</span>}</div>
                                        <div style={{ color: "var(--text-secondary)" }}>{s.address?.road || ""} {s.address?.town || s.address?.village || s.address?.city || s.address?.suburb || ""}</div>
                                    </div>
                                    <div style={{ textAlign: "right" }}>
                                        <div style={{ fontWeight: 800 }}>{s.distanceMeters ? `${(s.distanceMeters/1000).toFixed(2)} km` : ""}</div>
                                        <a 
                                            className="map-link-btn" 
                                            // UPDATED: Using a query string that includes the shop name and coordinates.
                                            href={`https://www.google.com/maps/search/?api=1&query=${mapQuery}`} 
                                            target="_blank" 
                                            rel="noreferrer"
                                        >Directions →</a>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </>
            )}

            {!loading && shops.length === 0 && !error && (
                <div className="empty-state" style={{ marginTop: 12 }}>
                    <p className="text-secondary">No seed/fertilizer shops found yet. Try a broader search term like "agro" or use the GPS button.</p>
                </div>
            )}
        </div>
    </div>
  );
}