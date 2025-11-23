import React, { useState, useEffect, useRef } from 'react';
import { 
    Package, PlusCircle, AlertTriangle, Trash2, Clock, Loader2, Database
} from 'lucide-react';

// --- Local Storage Keys (Persistence without Firebase) ---
const INVENTORY_KEY = 'agriInventory';
const HISTORY_KEY = 'agriInventoryHistory';
// ---------------------------------------------------------

// Helper function to safely parse localStorage JSON
const getInitialState = (key, defaultValue) => {
    try {
        const stored = localStorage.getItem(key);
        if (stored) return JSON.parse(stored);
    } catch (e) {
        console.error(`Error loading ${key} from localStorage:`, e);
    }
    return defaultValue;
};


function InventoryView() {
    const [inventory, setInventory] = useState(() => getInitialState(INVENTORY_KEY, []));
    const [history, setHistory] = useState(() => getInitialState(HISTORY_KEY, []));
    const [loading, setLoading] = useState(false); // No async loading needed for localStorage
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isHistoryOpen, setIsHistoryOpen] = useState(false);
    const [newItem, setNewItem] = useState({ type: 'Seed', name: '', quantity: '', unit: 'kg' });
    
    // --- Persistence Effects (Saves state to localStorage) ---

    // Save inventory whenever it changes
    useEffect(() => {
        try {
            localStorage.setItem(INVENTORY_KEY, JSON.stringify(inventory));
        } catch (e) {
            console.error("Error saving inventory to localStorage:", e);
        }
    }, [inventory]);

    // Save history whenever it changes
    useEffect(() => {
        try {
            // Convert JS Dates back to strings for reliable storage
            const serializableHistory = history.map(record => ({
                ...record,
                timestamp: record.timestamp ? record.timestamp.toISOString() : null
            }));
            localStorage.setItem(HISTORY_KEY, JSON.stringify(serializableHistory));
        } catch (e) {
            console.error("Error saving history to localStorage:", e);
        }
    }, [history]);

    // Helper to log actions
    const logHistoryAction = (action, item) => {
        const newRecord = {
            id: crypto.randomUUID(),
            action: action, // 'ADDED' or 'DELETED'
            item: { 
                name: item.name, 
                type: item.type, 
                quantity: item.quantity, 
                unit: item.unit 
            },
            timestamp: new Date(), // Use JS Date object
        };
        
        setHistory(prev => {
            const updatedHistory = [newRecord, ...prev];
            // Sort in memory to ensure new entry is at the top
            updatedHistory.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
            return updatedHistory;
        });
    };

    // CRUD: Add Item
    const handleAddItem = (e) => {
        e.preventDefault();
        if (!newItem.name.trim() || !newItem.quantity) return;

        const itemData = {
            id: crypto.randomUUID(), 
            ...newItem,
            quantity: Number(newItem.quantity),
            status: Number(newItem.quantity) < 10 ? 'Critical' : Number(newItem.quantity) < 50 ? 'Low' : 'Optimal',
        };

        setInventory(prev => {
            logHistoryAction('ADDED', itemData);
            return [...prev, itemData];
        });
        
        setIsModalOpen(false);
        setNewItem({ type: 'Seed', name: '', quantity: '', unit: 'kg' });
    };

    // CRUD: Delete Item
    const handleDelete = (item) => {
        setInventory(prev => {
            logHistoryAction('DELETED', item);
            return prev.filter(i => i.id !== item.id);
        });
    };

    const getStatusTheme = (status) => {
        switch (status) {
            case 'Critical': return { bg: '#fee2e2', text: '#ef4444', icon: <AlertTriangle size={18} /> };
            case 'Low': return { bg: '#fef3c7', text: '#f59e0b', icon: <AlertTriangle size={18} /> };
            default: return { bg: '#dcfce7', text: '#10b981', icon: <Package size={18} /> };
        }
    };

    const lowStockCount = inventory.filter(i => i.status !== 'Optimal').length;
    
    return (
        <div className="dashboard-container">
            <header className="content-header">
                <div>
                    <h1><Package /> Seed & Fertilizer Inventory</h1>
                    <p className="subtitle">Current stock levels for essential farm inputs (Data saved locally)</p>
                </div>
                <div style={{display: 'flex', gap: 10}}>
                    <button className="refresh-btn" onClick={() => setIsHistoryOpen(true)}>
                        <Clock size={18} /> History
                    </button>
                    <button className="refresh-btn primary-btn" onClick={() => setIsModalOpen(true)}>
                        <PlusCircle size={18} /> Add Item
                    </button>
                </div>
            </header>
            
            <p style={{fontSize: 10, color: '#94a3b8', textAlign: 'right'}}>Data is stored in your browser's local storage.</p>

            {lowStockCount > 0 && (
                 <div className="card danger-theme" style={{ marginBottom: 16 }}>
                    <div className="action-header">
                        <AlertTriangle size={20} />
                        <p className="card-label">STOCK ALERT</p>
                    </div>
                    <p>You have **{lowStockCount} items** in Low or Critical stock. Please re-order soon.</p>
                </div>
            )}

            <div className="inventory-grid">
                {inventory.length === 0 ? (
                    <div className="card" style={{ gridColumn: '1 / -1', textAlign: 'center' }}>
                        <p className="text-secondary">No items in inventory. Use 'Add Item' to start tracking your stock.</p>
                    </div>
                ) : (
                    inventory.map(item => {
                        const theme = getStatusTheme(item.status);
                        return (
                            <div key={item.id} className="inventory-card" style={{ borderLeftColor: theme.text }}>
                                <div className="card-content">
                                    <div className="item-name">{item.name}</div>
                                    <div className="item-type">{item.type}</div>
                                </div>
                                <div className="card-footer">
                                    <div className="item-quantity">
                                        <strong>{item.quantity}</strong> {item.unit}
                                    </div>
                                    <div className="item-status" style={{ backgroundColor: theme.bg, color: theme.text }}>
                                        {theme.icon} {item.status}
                                    </div>
                                    <button className="delete-btn" onClick={() => handleDelete(item)}>
                                        <Trash2 size={16} color="#94a3b8" />
                                    </button>
                                </div>
                            </div>
                        );
                    })
                )}
            </div>

            {/* Modal for Adding Item */}
            {isModalOpen && (
                <div className="modal-overlay">
                    <div className="modal-content">
                        <h3>Add New Inventory Item</h3>
                        <form onSubmit={handleAddItem}>
                            <div className="form-group">
                                <label>Type</label>
                                <select 
                                    value={newItem.type} 
                                    onChange={(e) => setNewItem({...newItem, type: e.target.value})}
                                    required
                                >
                                    <option>Seed</option>
                                    <option>Fertilizer</option>
                                    <option>Pesticide</option>
                                    <option>Other</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label>Name / Variety</label>
                                <input 
                                    type="text" 
                                    value={newItem.name} 
                                    onChange={(e) => setNewItem({...newItem, name: e.target.value})}
                                    placeholder="e.g., NPK 19:19:19 or Hybrid Corn"
                                    required
                                />
                            </div>
                            <div className="form-group-inline">
                                <div className="form-group">
                                    <label>Quantity</label>
                                    <input 
                                        type="number" 
                                        value={newItem.quantity} 
                                        onChange={(e) => setNewItem({...newItem, quantity: e.target.value})}
                                        placeholder="50"
                                        min="0"
                                        required
                                    />
                                </div>
                                <div className="form-group">
                                    <label>Unit</label>
                                    <select 
                                        value={newItem.unit} 
                                        onChange={(e) => setNewItem({...newItem, unit: e.target.value})}
                                    >
                                        <option>kg</option>
                                        <option>grams</option>
                                        <option>litres</option>
                                        <option>ml</option>
                                        <option>packs</option>
                                    </select>
                                </div>
                            </div>
                            <div className="modal-actions">
                                <button type="button" onClick={() => setIsModalOpen(false)} className="cancel-btn">Cancel</button>
                                <button type="submit" className="primary-btn">Save Item</button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
            
            {/* Modal for History */}
            {isHistoryOpen && (
                <div className="modal-overlay">
                    <div className="modal-content history-modal">
                        <h3 style={{marginBottom: 20}}><Clock size={20} style={{marginRight: 8}}/> Inventory History</h3>
                        <div className="history-list">
                            {history.length === 0 ? (
                                <p className="text-secondary" style={{textAlign: 'center'}}>No recent activity recorded.</p>
                            ) : (
                                history.map((record, index) => (
                                    <div key={record.id} className={`history-item ${record.action.toLowerCase()}`}>
                                        <span className="history-action">{record.action === 'ADDED' ? '➕ Added' : '🗑️ Removed'}</span>
                                        <span className="history-details">
                                            {record.item.quantity} {record.item.unit} of **{record.item.name}** ({record.item.type})
                                        </span>
                                        <span className="history-timestamp">
                                            {/* Attempt to display stored date, falling back to string if not a Date object */}
                                            {record.timestamp ? new Date(record.timestamp).toLocaleString('en-US', { hour: '2-digit', minute: '2-digit', month: 'short', day: 'numeric' }) : '...'}
                                        </span>
                                    </div>
                                ))
                            )}
                        </div>
                        <div className="modal-actions">
                            <button type="button" onClick={() => setIsHistoryOpen(false)} className="cancel-btn">Close</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default InventoryView;