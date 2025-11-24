// ==========================================
// 1. REACT FRONTEND (App.jsx)
// ==========================================
import { useState, useEffect, useRef } from 'react';
import { 
  CloudRain, Droplets, Thermometer, Gauge, Sprout, RefreshCw, WifiOff, 
  CheckCircle2, AlertTriangle, LayoutDashboard, MessageSquare, Menu, X, Send, User,
  Package // Icon for Inventory
} from 'lucide-react';
import './App.css';
import InventoryView from './InventoryView';
import SeedShopFinder from './SeedShopFinder'; // <-- Imported for the new 'shop' tab

// --- UPDATED: Use Relative Paths for Vercel Backend Connection ---
const API_URL = "/api/agri-advisory"; // Vercel routes this to agri-weather-web/api/app.py
const CHAT_URL = "/api/chat";         // Vercel routes this to agri-weather-web/api/app.py
// ----------------------------------------------------------------

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard'); 
  const [isSidebarOpen, setSidebarOpen] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      // NOTE: Using window.location.origin is generally not needed for relative paths,
      // but fetch(API_URL) will automatically resolve correctly on Vercel.
      const response = await fetch(API_URL); 
      if (!response.ok) throw new Error("Backend Offline (Status: " + response.status + ")");
      const json = await response.json();
      setData(json);
    } catch (err) {
      console.error(err);
      setError("Could not connect to Python Backend. Check Vercel logs for function errors.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    // Polls every 30 seconds for new sensor data
    const interval = setInterval(fetchData, 30000); 
    return () => clearInterval(interval);
  }, []);

  const toggleSidebar = () => setSidebarOpen(!isSidebarOpen);

  if (loading && !data) return <div className="loading-container"><div className="spinner"></div><p>CONTACTING SATELLITE...</p></div>;
  if (error) return <div className="error-container"><WifiOff size={64} color="#ef4444" /><h2>Connection Lost</h2><p>{error}</p><button onClick={fetchData} className="retry-btn">RETRY</button></div>;

  return (
    <div className="app-layout">
      <div className="mobile-header">
        <button onClick={toggleSidebar} className="menu-btn"><Menu /></button>
        <h1>Agri-AI</h1>
        <span className="badge location">{data.location}</span>
      </div>

      <aside className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h2>Agri-AI</h2>
          <button onClick={toggleSidebar} className="close-btn"><X /></button>
        </div>
        
        <nav className="nav-menu">
          <button className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => { setActiveTab('dashboard'); setSidebarOpen(false); }}>
            <LayoutDashboard size={20} /><span>Dashboard</span>
          </button>
          <button className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => { setActiveTab('chat'); setSidebarOpen(false); }}>
            <MessageSquare size={20} /><span>Agronomist Chat</span>
          </button>
          {/* Inventory Tab */}
          <button className={`nav-item ${activeTab === 'inventory' ? 'active' : ''}`} onClick={() => { setActiveTab('inventory'); setSidebarOpen(false); }}>
            <Package size={20} /><span>Inventory</span>
          </button>
          {/* Shop Finder Tab */}
          <button className={`nav-item ${activeTab === 'shop' ? 'active' : ''}`} onClick={() => { setActiveTab('shop'); setSidebarOpen(false); }}>
            <Sprout size={20} /><span>Shop Finder</span>
          </button>
        </nav>

        <div className="sidebar-footer">
          <p className="status-dot"><span className="dot-online"></span> System Online</p>
          <p className="timestamp">Updated: {data.timestamp.split(' ')[1]}</p>
        </div>
      </aside>

      <main className="main-content">
        <div className="dashboard-container"> {/* Container for padding and max-width */}
          {activeTab === 'dashboard' ? (
            <DashboardView data={data} refresh={fetchData} switchToChat={() => setActiveTab('chat')} />
          ) : activeTab === 'chat' ? (
            <ChatView initialData={data} />
          ) : activeTab === 'inventory' ? (
            <InventoryView />
          ) : ( // Renders SeedShopFinder for the 'shop' tab
            <SeedShopFinder />
          )}
        </div>
      </main>
      
      {isSidebarOpen && <div className="overlay" onClick={toggleSidebar}></div>}
    </div>
  );
}

function DashboardView({ data, refresh, switchToChat }) {
  const isRain = data.analysis.forecast.includes('RAIN');
  const isStop = data.analysis.action.includes('STOP') || data.analysis.action.includes('DELAY') || data.analysis.action.includes('NO ACTION');
  const soilMoistureLow = data.sensors.soil_moisture < 40;


  return (
    <div className="dashboard-container">
      <header className="content-header">
        <div><h1>Farm Overview</h1><p className="subtitle">{data.season}</p></div>
        <button onClick={refresh} className="refresh-btn"><RefreshCw size={20} /></button>
      </header>

      <div className="dashboard-grid">
        <div className={`card verdict-card ${isRain ? 'rain-theme' : 'dry-theme'}`}>
          <p className="card-label">Physics Model Forecast</p>
          <h2>{data.analysis.forecast}</h2>
          <div className="confidence">{data.analysis.confidence}% Confidence</div>
        </div>

        <div className={`card action-card ${soilMoistureLow ? 'danger-theme' : 'safe-theme'}`}>
          <div className="action-header">
            {soilMoistureLow ? <AlertTriangle size={24} /> : <CheckCircle2 size={24} />}
            <p className="card-label">REQUIRED ACTION</p>
          </div>
          <h3>{data.analysis.action}</h3>
          <p className="reason">{data.analysis.reason}</p>
        </div>

        <div className="sensors-section">
          <h3 className="section-title">Live Sensors</h3>
          <div className="sensors-grid">
            <SensorCard label="Temp" value={data.sensors.temperature} unit="°C" icon={<Thermometer size={18} />} />
            <SensorCard label="Humidity" value={data.sensors.humidity} unit="%" icon={<Droplets size={18} />} color="#2563eb" />
            <SensorCard 
              label="Soil" 
              value={data.sensors.soil_moisture} 
              unit="%" 
              icon={<Sprout size={18} />} 
              color={data.sensors.soil_moisture < 40 ? "#dc2626" : "#16a34a"} 
            />
            <SensorCard label="Pressure" value={data.sensors.pressure} unit="hPa" icon={<Gauge size={18} />} />
          </div>
        </div>

        <div className="card chat-teaser" onClick={switchToChat}>
          <div className="teaser-content">
            <div className="icon-box"><MessageSquare size={24} color="white" /></div>
            <div><h4>Ask the Agronomist</h4><p>Tap to chat about crops & weather</p></div>
          </div>
          <div className="arrow">→</div>
        </div>
      </div>
    </div>
  );
}

function ChatView({ initialData }) {
  const [messages, setMessages] = useState([
    { 
      id: 1, 
      sender: 'system', 
      text: `Hello! I am monitoring ${initialData.location}. Soil moisture is ${initialData.sensors.soil_moisture}%.` 
    },
    { 
      id: 2, 
      sender: 'ai', 
      text: initialData.llm_advisory.replace(/\*\*/g, '') 
    }
  ]);
  const [inputText, setInputText] = useState("");
  const [isSending, setIsSending] = useState(false);
  const chatBoxRef = useRef(null);

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMsg = { id: Date.now(), sender: 'user', text: inputText };
    
    // Prepare history to send to backend (excluding system/initial messages)
    const historyForBackend = messages
      .filter(msg => msg.sender !== 'system')
      .map(msg => ({ sender: msg.sender, text: msg.text }));

    setMessages(prev => [...prev, userMsg]);
    setInputText("");
    setIsSending(true);

    try {
      // Use relative CHAT_URL
      const response = await fetch(CHAT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: userMsg.text,
          district: initialData.location,
          // History is currently NOT supported by the Vercel Serverless chat endpoint
          // history: historyForBackend 
        })
      });
      
      const json = await response.json();
      
      if (!response.ok) {
        throw new Error(json.detail || "Server Error");
      }

      const replyText = json.reply && json.reply.trim() !== "" ? json.reply : "I'm having trouble analyzing that right now. Please try again.";
      
      const aiMsg = { id: Date.now() + 1, sender: 'ai', text: replyText };
      setMessages(prev => [...prev, aiMsg]);
      
    } catch (err) {
      console.error("Chat Error:", err);
      setMessages(prev => [...prev, { id: Date.now(), sender: 'system', text: `Error: ${err.message || "Could not reach Agronomist."}` }]);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className="chat-container">
      <header className="content-header">
        <h1>Agronomist AI</h1>
      </header>

      <div className="chat-box" ref={chatBoxRef}>
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.sender}`}>
            <div className="avatar">
              {msg.sender === 'ai' ? '👨‍🌾' : msg.sender === 'user' ? <User size={20} /> : '🤖'}
            </div>
            <div className={`bubble ${msg.sender}-bubble`}>
              {msg.sender === 'ai' && <p className="bubble-header">AGRONOMIST</p>}
              <div className="advisory-text">{msg.text}</div>
            </div>
          </div>
        ))}
        {isSending && (
          <div className="message ai">
            <div className="avatar">👨‍🌾</div>
            <div className="bubble ai-bubble">
              <div className="typing-indicator">Thinking...</div>
            </div>
          </div>
        )}
      </div>

      <div className="chat-input-area">
        <input 
          type="text" 
          placeholder="Ask about crops, irrigation, or pests..." 
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          disabled={isSending}
        />
        <button className="send-btn" onClick={sendMessage} disabled={isSending || !inputText.trim()}>
          <Send size={20} />
        </button>
      </div>
    </div>
  );
}

function SensorCard({ label, value, unit, icon, color = "#1e293b" }) {
  return (
    <div className="sensor-card">
      <div className="sensor-header"><span className="sensor-label">{label}</span>{icon}</div>
      <div className="sensor-value-box">
        <span className="sensor-value" style={{ color }}>{value}</span>
        <span className="sensor-unit">{unit}</span>
      </div>
    </div>
  );
}

export default App;