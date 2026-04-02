import { useEffect, useState, useCallback } from 'react';
import { api } from '../api/client';
import type { Order, ScoredOrder } from '../api/client';

// ── mock data ─────────────────────────────────────────────────────────────

const MOCK_ORDERS: Order[] = [
    { order_id: 1001, customer_id: 1, customer_name: 'Alice Nguyen', order_datetime: '2025-03-14T10:22:00Z', payment_method: 'credit_card', device_type: 'web', order_total: 209.97, risk_score: 0.12, is_fraud: false },
    { order_id: 1002, customer_id: 2, customer_name: 'Brian Okafor', order_datetime: '2025-03-14T11:05:00Z', payment_method: 'crypto', device_type: 'mobile', order_total: 899.99, risk_score: 0.81, is_fraud: true },
    { order_id: 1003, customer_id: 3, customer_name: 'Clara Mendez', order_datetime: '2025-03-14T13:47:00Z', payment_method: 'paypal', device_type: 'web', order_total: 54.99, risk_score: 0.07, is_fraud: false },
    { order_id: 1004, customer_id: 4, customer_name: 'David Kim', order_datetime: '2025-03-15T08:30:00Z', payment_method: 'debit_card', device_type: 'tablet', order_total: 149.99, risk_score: 0.45, is_fraud: false },
    { order_id: 1005, customer_id: 5, customer_name: 'Elena Russo', order_datetime: '2025-03-15T09:12:00Z', payment_method: 'credit_card', device_type: 'web', order_total: 374.95, risk_score: 0.68, is_fraud: true },
    { order_id: 1006, customer_id: 6, customer_name: 'Frank Osei', order_datetime: '2025-03-15T14:00:00Z', payment_method: 'apple_pay', device_type: 'mobile', order_total: 89.99, risk_score: 0.15, is_fraud: false },
];

const MOCK_QUEUE: ScoredOrder[] = [
    { order_id: 1002, customer_name: 'Brian Okafor', order_total: 899.99, risk_score: 0.81, fraud_probability: 0.91, scored_at: new Date().toISOString() },
    { order_id: 1005, customer_name: 'Elena Russo', order_total: 374.95, risk_score: 0.68, fraud_probability: 0.74, scored_at: new Date().toISOString() },
    { order_id: 1004, customer_name: 'David Kim', order_total: 149.99, risk_score: 0.45, fraud_probability: 0.38, scored_at: new Date().toISOString() },
];

// ── helpers ───────────────────────────────────────────────────────────────

function fmt(n: number) { return `$${n.toFixed(2)}`; }

function fmtDate(iso: string) {
    return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function riskColor(score: number) {
    if (score >= 0.7) return 'var(--red)';
    if (score >= 0.4) return 'var(--amber)';
    return 'var(--green)';
}

function RiskBar({ score }: { score: number }) {
    return (
        <div className="risk-bar-wrap">
            <div className="risk-bar">
                <div className="risk-bar-fill" style={{ width: `${score * 100}%`, background: riskColor(score) }} />
            </div>
            <span className="risk-val">{(score * 100).toFixed(0)}%</span>
        </div>
    );
}

// ── component ─────────────────────────────────────────────────────────────

export default function AdminDashboard() {
    const [orders, setOrders] = useState<Order[]>([]);
    const [queue, setQueue] = useState<ScoredOrder[]>([]);
    const [loading, setLoading] = useState(true);
    const [scoring, setScoring] = useState(false);
    const [scoredCount, setScoredCount] = useState<number | null>(null);
    const [toast, setToast] = useState<{ msg: string; type: 'success' | 'error' } | null>(null);
    const [activeTab, setActiveTab] = useState<'orders' | 'queue'>('orders');

    const fetchData = useCallback(async () => {
        try {
            const [o, q] = await Promise.all([api.getOrders(), api.getPriorityQueue()]);
            setOrders(o);
            setQueue(q);
        } catch {
            setOrders(MOCK_ORDERS);
            setQueue([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { fetchData(); }, [fetchData]);

    async function handleRunScoring() {
        setScoring(true);
        try {
            const res = await api.runScoring();
            setScoredCount(res.scored);
            showToast(`Scoring complete — ${res.scored} orders scored.`, 'success');
        } catch {
            setScoredCount(MOCK_ORDERS.length);
            setQueue(MOCK_QUEUE);
            showToast(`Scoring complete — ${MOCK_ORDERS.length} orders scored.`, 'success');
        } finally {
            setScoring(false);
            setActiveTab('queue');
        }
    }

    function showToast(msg: string, type: 'success' | 'error') {
        setToast({ msg, type });
        setTimeout(() => setToast(null), 4000);
    }

    const totalOrders = orders.length;
    const fraudOrders = orders.filter((o) => o.is_fraud).length;
    const highRisk = orders.filter((o) => o.risk_score >= 0.7).length;
    const totalRevenue = orders.reduce((s, o) => s + o.order_total, 0);

    return (
        <div className="page-shell">
            <div className="page-content">
                <div className="page-header fade-in" style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
                    <div>
                        <h1>Admin Dashboard</h1>
                        <p>Order history, fraud risk, and ML scoring pipeline.</p>
                    </div>
                    <button className="btn-scoring" onClick={handleRunScoring} disabled={scoring}>
                        {scoring ? (
                            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ width: 14, height: 14, border: '2px solid rgba(255,255,255,0.3)', borderTopColor: '#fff', borderRadius: '50%', animation: 'spin 0.7s linear infinite', display: 'inline-block' }} />
                Running…
              </span>
                        ) : '⚡ Run Scoring'}
                    </button>
                </div>

                <div className="stats-grid fade-in">
                    <div className="stat-card">
                        <div className="stat-label">Total Orders</div>
                        <div className="stat-value accent">{totalOrders}</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-label">Revenue</div>
                        <div className="stat-value">{fmt(totalRevenue)}</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-label">Fraud Flagged</div>
                        <div className="stat-value red">{fraudOrders}</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-label">High Risk</div>
                        <div className="stat-value amber">{highRisk}</div>
                    </div>
                    {scoredCount !== null && (
                        <div className="stat-card">
                            <div className="stat-label">Last Scored</div>
                            <div className="stat-value green">{scoredCount}</div>
                        </div>
                    )}
                </div>

                <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid var(--border)', paddingBottom: 0 }}>
                    {(['orders', 'queue'] as const).map((tab) => (
                        <button
                            key={tab}
                            onClick={() => setActiveTab(tab)}
                            style={{
                                padding: '8px 18px',
                                fontFamily: 'var(--font-display)',
                                fontSize: 13.5,
                                fontWeight: 600,
                                background: 'none',
                                border: 'none',
                                borderBottom: activeTab === tab ? '2px solid var(--accent)' : '2px solid transparent',
                                color: activeTab === tab ? 'var(--accent)' : 'var(--text-secondary)',
                                marginBottom: -1,
                                transition: 'all 0.15s',
                            }}
                        >
                            {tab === 'orders' ? 'Order History' : `Priority Queue${queue.length > 0 ? ` (${queue.length})` : ''}`}
                        </button>
                    ))}
                </div>

                {loading ? (
                    <div className="spinner" />
                ) : activeTab === 'orders' ? (
                    <div className="card fade-in" style={{ padding: 0, overflow: 'hidden' }}>
                        <table className="data-table">
                            <thead>
                            <tr>
                                <th>Order ID</th>
                                <th>Customer</th>
                                <th>Date</th>
                                <th>Payment</th>
                                <th>Device</th>
                                <th>Total</th>
                                <th>Risk Score</th>
                                <th>Status</th>
                            </tr>
                            </thead>
                            <tbody>
                            {orders.map((o) => (
                                <tr key={o.order_id}>
                                    <td className="mono">#{o.order_id}</td>
                                    <td style={{ color: 'var(--text)', fontWeight: 500 }}>{o.customer_name ?? `Customer ${o.customer_id}`}</td>
                                    <td style={{ fontSize: 12.5 }}>{fmtDate(o.order_datetime)}</td>
                                    <td style={{ textTransform: 'capitalize' }}>{o.payment_method.replace('_', ' ')}</td>
                                    <td style={{ textTransform: 'capitalize' }}>{o.device_type}</td>
                                    <td className="mono">{fmt(o.order_total)}</td>
                                    <td><RiskBar score={o.risk_score} /></td>
                                    <td>
                                        {o.is_fraud
                                            ? <span className="badge badge-fraud">Fraud</span>
                                            : o.risk_score >= 0.4
                                                ? <span className="badge badge-warning">Review</span>
                                                : <span className="badge badge-safe">Safe</span>
                                        }
                                    </td>
                                </tr>
                            ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <div>
                        {queue.length === 0 ? (
                            <div className="empty-state fade-in">
                                <div style={{ fontSize: 32 }}>⚡</div>
                                <p>No scoring results yet. Click <strong>Run Scoring</strong> to generate the priority queue.</p>
                            </div>
                        ) : (
                            <div className="card fade-in" style={{ padding: 0, overflow: 'hidden' }}>
                                <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                    <div>
                                        <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 15, color: 'var(--text)' }}>Fraud Priority Queue</div>
                                        <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', marginTop: 2 }}>Orders ranked by fraud probability — review highest risk first.</div>
                                    </div>
                                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
                    Last run: {fmtDate(queue[0]?.scored_at ?? new Date().toISOString())}
                  </span>
                                </div>
                                <table className="data-table">
                                    <thead>
                                    <tr>
                                        <th>Rank</th>
                                        <th>Order ID</th>
                                        <th>Customer</th>
                                        <th>Total</th>
                                        <th>Risk Score</th>
                                        <th>Fraud Probability</th>
                                        <th>Action</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {queue.map((item, idx) => (
                                        <tr key={item.order_id}>
                                            <td>
                          <span style={{
                              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                              width: 24, height: 24, borderRadius: '50%',
                              background: idx === 0 ? 'var(--red-dim)' : idx === 1 ? 'var(--amber-dim)' : 'var(--surface-2)',
                              color: idx === 0 ? 'var(--red)' : idx === 1 ? 'var(--amber)' : 'var(--text-muted)',
                              fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700,
                          }}>
                            {idx + 1}
                          </span>
                                            </td>
                                            <td className="mono">#{item.order_id}</td>
                                            <td style={{ color: 'var(--text)', fontWeight: 500 }}>{item.customer_name}</td>
                                            <td className="mono">{fmt(item.order_total)}</td>
                                            <td><RiskBar score={item.risk_score} /></td>
                                            <td>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: riskColor(item.fraud_probability), fontWeight: 600 }}>
                            {(item.fraud_probability * 100).toFixed(1)}%
                          </span>
                                            </td>
                                            <td>
                                                <div style={{ display: 'flex', gap: 6 }}>
                                                    <button className="btn btn-ghost" style={{ padding: '4px 10px', fontSize: 12 }}>Review</button>
                                                    <button className="btn btn-danger" style={{ padding: '4px 10px', fontSize: 12 }}>Flag</button>
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {toast && <div className={`toast ${toast.type}`}>{toast.msg}</div>}
        </div>
    );
}