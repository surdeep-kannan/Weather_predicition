import React, { useState, useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import { RefreshCw, AlertTriangle, Navigation, Sprout } from "lucide-react";
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
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

// basic agri filter
function isAgriShop(item) {
  const name = (item.name || item.display_name || "").toLowerCase();
  const addr = JSON.stringify(item.address || "").toLowerCase();
  const tags = JSON.stringify(item.extratags || item.tags || {}).toLowerCase();

  return ALLOWED_KEYWORDS.some(k => name.includes(k) || addr.includes(k) || tags.includes(k));
}

function Recenter({ lat, lon }) {
  const map = useMap();
  useEffect(() => {
    if (lat && lon) map.setView([lat, lon], 13);
  }, [lat, lon, map]);
  return null;
}

export default function SeedShopFinder() {
  const [location, setLocation] = useState(null);
  const [shops, setShops] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // wrapper to search with multiple strategies
  const searchPipeline = async (loc) => {
    setLoading(true);
    setError("");
    try {
      let all = [];

      // 1) Nearby (short)
      try {
        const urlN = `https://us1.locationiq.com/v1/nearby?key=${LOCATIONIQ_API_KEY}&lat=${loc.lat}&lon=${loc.lon}&radius=6000&format=json`;
        const r = await fetch(urlN);
        if (r.ok) {
          const d = await r.json();
          all = all.concat(d || []);
        }
      } catch (e) { /* ignore */ }

      // 2) Expanded nearby (25km)
      try {
        const urlE = `https://us1.locationiq.com/v1/nearby?key=${LOCATIONIQ_API_KEY}&lat=${loc.lat}&lon=${loc.lon}&radius=25000&format=json`;
        const r2 = await fetch(urlE);
        if (r2.ok) {
          const d2 = await r2.json();
          all = all.concat(d2 || []);
        }
      } catch (e) { /* ignore */ }

      // 3) Forward searches (text)
      const SEARCH_TERMS = [
        "seed shop", "fertilizer shop", "pesticide shop",
        "agro centre", "agri input store", "nursery", "krishi kendra"
      ];
      const forwardPromises = SEARCH_TERMS.map(term => {
        const q = encodeURIComponent(`${term} near ${loc.lat},${loc.lon}`);
        const url = `https://us1.locationiq.com/v1/search?key=${LOCATIONIQ_API_KEY}&q=${q}&format=json&limit=8`;
        return fetch(url).then(res => res.ok ? res.json() : []).catch(()=>[]);
      });
      const settled = await Promise.allSettled(forwardPromises);
      settled.forEach(s => {
        if (s.status === 'fulfilled' && Array.isArray(s.value)) all = all.concat(s.value);
      });

      // normalize lat/lon and filter agri shops
      const normalized = all
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

      // dedupe by rounded coords
      const map = new Map();
      normalized.forEach(it => {
        const k = `${it.lat.toFixed(5)}|${it.lon.toFixed(5)}`;
        if (!map.has(k)) map.set(k, it);
      });
      const unique = Array.from(map.values());

      // attach distance and sort
      const withDist = unique.map(it => ({ 
        ...it, 
        distanceMeters: haversine(loc.lat, loc.lon, it.lat, it.lon)
      })).sort((a,b) => a.distanceMeters - b.distanceMeters);

      setShops(withDist.slice(0, 60));
      if (withDist.length === 0) setError("No seed/fertilizer shops found nearby. Try refreshing or searching a nearby town.");
    } catch (err) {
      console.error(err);
      setError("Search failed. Check API key or network.");
    } finally {
      setLoading(false);
    }
  };

  // get GPS & start search automatically
  const getAndSearch = () => {
    setError("");
    setShops([]);
    setLoading(true);

    if (!navigator.geolocation) {
      setError("Geolocation not supported.");
      setLoading(false);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      pos => {
        const loc = { lat: pos.coords.latitude, lon: pos.coords.longitude };
        setLocation(loc);
        searchPipeline(loc).finally(()=>setLoading(false));
      },
      err => {
        setError("GPS error: " + err.message);
        setLoading(false);
      },
      { enableHighAccuracy: true, timeout: 15000 }
    );
  };

  useEffect(() => {
    getAndSearch();
  }, []);

  return (
    <div className="dashboard-container">
      <div className="content-header">
        <div>
          <h1><Sprout /> Seed Shop Finder</h1>
          <p className="subtitle">Automatically finds seed / fertilizer / nursery shops near you</p>
        </div>

        <button className="refresh-btn" onClick={getAndSearch} disabled={loading}>
          <RefreshCw size={18} /> Refresh
        </button>
      </div>

      {error && (
        <div className="card danger-theme" style={{ marginBottom: 16 }}>
          <div className="action-header">
            <AlertTriangle size={20} />
            <p className="card-label">Problem</p>
          </div>
          <p>{error}</p>
        </div>
      )}

      <div className="card" style={{ padding: 0, height: "68vh", overflow: "hidden" }}>
        <MapContainer
          center={location ? [location.lat, location.lon] : [20.5937, 78.9629]}
          zoom={location ? 13 : 5}
          style={{ height: "100%", width: "100%" }}
        >
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

          {location && (
            <>
              <Marker position={[location.lat, location.lon]}>
                <Popup>You are here</Popup>
              </Marker>
              <Recenter lat={location?.lat} lon={location?.lon} />
            </>
          )}

          {shops.map((s, i) => (
            <Marker key={i} position={[s.lat, s.lon]}>
              <Popup>
                <div style={{ minWidth: 180 }}>
                  <strong>{s.name || s.display_name || "Agri Shop"}</strong>
                  <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
                    {s.address?.road ? s.address.road + ", " : ""}
                    {s.address?.town || s.address?.village || s.address?.city || ""}
                  </div>
                  <div style={{ marginTop: 8 }}>
                    <a href={`https://www.google.com/maps/search/?api=1&query=${s.lat},${s.lon}`} target="_blank" rel="noreferrer">
                      <Navigation size={14} /> Open in Maps
                    </a>
                    <div style={{ fontSize: 12, marginTop: 6, color: "var(--text-secondary)" }}>
                      {s.distanceMeters ? `${(s.distanceMeters/1000).toFixed(2)} km` : ""}
                    </div>
                  </div>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>

      <div style={{ marginTop: 12 }}>
        {loading && <p>Searching for shops...</p>}

        {!loading && shops.length > 0 && (
          <>
            <h3 className="section-title">Nearby results ({shops.length})</h3>
            <div style={{ display: "grid", gap: 10 }}>
              {shops.map((s, i) => (
                <div key={i} className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <div style={{ fontWeight: 800 }}>{s.name || s.display_name || "Agri Shop"}</div>
                    <div style={{ color: "var(--text-secondary)" }}>{s.address?.road || ""} {s.address?.town || s.address?.village || ""}</div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontWeight: 800 }}>{s.distanceMeters ? `${(s.distanceMeters/1000).toFixed(2)} km` : ""}</div>
                    <a className="map-link-btn" href={`https://www.google.com/maps/search/?api=1&query=${s.lat},${s.lon}`} target="_blank" rel="noreferrer">Directions →</a>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {!loading && shops.length === 0 && !error && (
          <div className="empty-state" style={{ marginTop: 12 }}>
            <p className="text-secondary">No seed/fertilizer shops found yet. Try refreshing or searching from a nearby town center.</p>
          </div>
        )}
      </div>
    </div>
  );
}
