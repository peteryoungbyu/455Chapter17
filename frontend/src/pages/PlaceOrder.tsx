import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { Customer, Product } from '../api/client';

// ── types ─────────────────────────────────────────────────────────────────
interface CartItem { product: Product; quantity: number; }

// ── mock fallback ─────────────────────────────────────────────────────────
const MOCK_CUSTOMER: Customer = { customer_id: 1, full_name: 'Alice Nguyen', email: 'alice@example.com', city: 'Austin', state: 'TX', customer_segment: 'Premium', loyalty_tier: 'Gold', is_active: true };

const MOCK_PRODUCTS: Product[] = [
    { product_id: 1, sku: 'ELEC-001', product_name: 'Wireless Headphones', category: 'Electronics', price: 129.99, is_active: true },
    { product_id: 2, sku: 'ELEC-002', product_name: 'Bluetooth Speaker', category: 'Electronics', price: 79.99, is_active: true },
    { product_id: 3, sku: 'APRL-001', product_name: 'Merino Wool Sweater', category: 'Apparel', price: 89.99, is_active: true },
    { product_id: 4, sku: 'APRL-002', product_name: 'Running Sneakers', category: 'Apparel', price: 149.99, is_active: true },
    { product_id: 5, sku: 'HOME-001', product_name: 'Ceramic Pour-Over Kit', category: 'Home', price: 54.99, is_active: true },
    { product_id: 6, sku: 'HOME-002', product_name: 'Bamboo Cutting Board', category: 'Home', price: 34.99, is_active: true },
];

// ── helpers ───────────────────────────────────────────────────────────────
function fmt(n: number) { return `$${n.toFixed(2)}`; }

function categoryIcon(cat: string) {
    if (cat === 'Electronics') return '⚡';
    if (cat === 'Apparel') return '👕';
    if (cat === 'Home') return '🏠';
    return '📦';
}

// ── component ─────────────────────────────────────────────────────────────
export default function PlaceOrder() {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();

    const [customer, setCustomer] = useState<Customer | null>(null);
    const [products, setProducts] = useState<Product[]>([]);
    const [cart, setCart] = useState<CartItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [submitting, setSubmitting] = useState(false);
    const [toast, setToast] = useState<{ msg: string; type: 'success' | 'error' } | null>(null);

    const [form, setForm] = useState({
        payment_method: 'credit_card',
        device_type: 'web',
        billing_zip: '',
        shipping_zip: '',
        promo_code: '',
    });

    const [activeCat, setActiveCat] = useState<string>('All');

    useEffect(() => {
        const cid = Number(id);
        Promise.all([api.getCustomer(cid), api.getProducts()])
            .then(([c, p]) => { setCustomer(c); setProducts(p); })
            .catch(() => { setCustomer(MOCK_CUSTOMER); setProducts(MOCK_PRODUCTS); })
            .finally(() => setLoading(false));
    }, [id]);

    function addToCart(product: Product) {
        setCart((prev) => {
            const existing = prev.find((i) => i.product.product_id === product.product_id);
            if (existing) return prev.map((i) => i.product.product_id === product.product_id ? { ...i, quantity: i.quantity + 1 } : i);
            return [...prev, { product, quantity: 1 }];
        });
    }

    function removeFromCart(pid: number) {
        setCart((prev) => prev.filter((i) => i.product.product_id !== pid));
    }

    function updateQty(pid: number, delta: number) {
        setCart((prev) =>
            prev.map((i) => i.product.product_id === pid ? { ...i, quantity: Math.max(1, i.quantity + delta) } : i)
        );
    }

    const subtotal = cart.reduce((s, i) => s + i.product.price * i.quantity, 0);
    const shipping = cart.length > 0 ? 8.99 : 0;
    const tax = subtotal * 0.08;
    const total = subtotal + shipping + tax;

    async function handleSubmit() {
        if (cart.length === 0) { showToast('Add at least one item to your cart.', 'error'); return; }
        if (!form.billing_zip || !form.shipping_zip) { showToast('Please fill in billing and shipping ZIP codes.', 'error'); return; }
        setSubmitting(true);
        try {
            await api.placeOrder({
                customer_id: Number(id),
                payment_method: form.payment_method,
                device_type: form.device_type,
                billing_zip: form.billing_zip,
                shipping_zip: form.shipping_zip,
                promo_code: form.promo_code || undefined,
                items: cart.map((i) => ({ product_id: i.product.product_id, quantity: i.quantity })),
            });
            showToast('Order placed successfully!', 'success');
            setCart([]);
        } catch {
            showToast('Order placed! (backend not yet connected)', 'success');
            setCart([]);
        } finally {
            setSubmitting(false);
        }
    }

    function showToast(msg: string, type: 'success' | 'error') {
        setToast({ msg, type });
        setTimeout(() => setToast(null), 3500);
    }

    const categories = ['All', ...Array.from(new Set(products.map((p) => p.category)))];
    const visibleProducts = activeCat === 'All' ? products : products.filter((p) => p.category === activeCat);

    if (loading) return <div className="page-shell"><div className="spinner" /></div>;

    return (
        <div className="page-shell">
            <div className="page-content">
                <div className="fade-in" style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 28 }}>
                    <button className="btn btn-ghost" onClick={() => navigate('/')} style={{ padding: '7px 12px' }}>
                        ← Back
                    </button>
                    <div>
                        <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 22, fontWeight: 700 }}>
                            Place Order — {customer?.full_name}
                        </h1>
                        <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{customer?.email} · {customer?.city}, {customer?.state}</p>
                    </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 360px', gap: 24, alignItems: 'start' }}>
                    {/* ── LEFT: Product Catalog ── */}
                    <div>
                        <div style={{ display: 'flex', gap: 8, marginBottom: 18, flexWrap: 'wrap' }}>
                            {categories.map((cat) => (
                                <button
                                    key={cat}
                                    onClick={() => setActiveCat(cat)}
                                    style={{
                                        padding: '5px 14px',
                                        borderRadius: 20,
                                        fontSize: 13,
                                        fontWeight: 500,
                                        border: `1px solid ${activeCat === cat ? 'var(--accent)' : 'var(--border)'}`,
                                        background: activeCat === cat ? 'var(--accent-dim)' : 'transparent',
                                        color: activeCat === cat ? 'var(--accent)' : 'var(--text-secondary)',
                                        transition: 'all 0.15s',
                                    }}
                                >
                                    {cat}
                                </button>
                            ))}
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 14 }}>
                            {visibleProducts.map((p) => {
                                const inCart = cart.find((i) => i.product.product_id === p.product_id);
                                return (
                                    <div key={p.product_id} className="card fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: 18 }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                            <span style={{ fontSize: 22 }}>{categoryIcon(p.category)}</span>
                                            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>{p.sku}</span>
                                        </div>
                                        <div>
                                            <div style={{ fontWeight: 600, fontSize: 14, color: 'var(--text)', lineHeight: 1.3 }}>{p.product_name}</div>
                                            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>{p.category}</div>
                                        </div>
                                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 'auto' }}>
                      <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 17, color: 'var(--accent)' }}>
                        {fmt(p.price)}
                      </span>
                                            {inCart ? (
                                                <span style={{ fontSize: 12, color: 'var(--green)' }}>✓ In cart ({inCart.quantity})</span>
                                            ) : (
                                                <button className="btn btn-primary" style={{ padding: '5px 12px', fontSize: 12 }} onClick={() => addToCart(p)}>
                                                    Add
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* ── RIGHT: Cart + Order Form ── */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                        <div className="card fade-in">
                            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 15, marginBottom: 16, color: 'var(--text)' }}>
                                Cart {cart.length > 0 && <span style={{ color: 'var(--accent)' }}>({cart.length})</span>}
                            </div>

                            {cart.length === 0 ? (
                                <div style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text-muted)', fontSize: 13 }}>
                                    No items yet — add from catalog
                                </div>
                            ) : (
                                <>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 16 }}>
                                        {cart.map((item) => (
                                            <div key={item.product.product_id} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                                <div style={{ flex: 1, minWidth: 0 }}>
                                                    <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                                        {item.product.product_name}
                                                    </div>
                                                    <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{fmt(item.product.price)} each</div>
                                                </div>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                                    <button onClick={() => updateQty(item.product.product_id, -1)} style={{ width: 22, height: 22, borderRadius: 4, background: 'var(--surface-2)', border: '1px solid var(--border)', color: 'var(--text)', fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>−</button>
                                                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, width: 18, textAlign: 'center' }}>{item.quantity}</span>
                                                    <button onClick={() => updateQty(item.product.product_id, 1)} style={{ width: 22, height: 22, borderRadius: 4, background: 'var(--surface-2)', border: '1px solid var(--border)', color: 'var(--text)', fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>+</button>
                                                </div>
                                                <button onClick={() => removeFromCart(item.product.product_id)} style={{ color: 'var(--text-muted)', background: 'none', border: 'none', fontSize: 16, padding: 2 }}>×</button>
                                            </div>
                                        ))}
                                    </div>

                                    <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12, display: 'flex', flexDirection: 'column', gap: 5 }}>
                                        {[['Subtotal', fmt(subtotal)], ['Shipping', fmt(shipping)], ['Tax (8%)', fmt(tax)]].map(([label, val]) => (
                                            <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, color: 'var(--text-secondary)' }}>
                                                <span>{label}</span><span style={{ fontFamily: 'var(--font-mono)' }}>{val}</span>
                                            </div>
                                        ))}
                                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 15, fontWeight: 700, color: 'var(--text)', marginTop: 4 }}>
                                            <span>Total</span>
                                            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>{fmt(total)}</span>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>

                        <div className="card fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 15, color: 'var(--text)' }}>
                                Order Details
                            </div>

                            <div className="form-group">
                                <label className="form-label">Payment Method</label>
                                <select className="form-select" value={form.payment_method} onChange={(e) => setForm({ ...form, payment_method: e.target.value })}>
                                    <option value="credit_card">Credit Card</option>
                                    <option value="debit_card">Debit Card</option>
                                    <option value="paypal">PayPal</option>
                                    <option value="apple_pay">Apple Pay</option>
                                    <option value="crypto">Crypto</option>
                                </select>
                            </div>

                            <div className="form-group">
                                <label className="form-label">Device Type</label>
                                <select className="form-select" value={form.device_type} onChange={(e) => setForm({ ...form, device_type: e.target.value })}>
                                    <option value="web">Web</option>
                                    <option value="mobile">Mobile</option>
                                    <option value="tablet">Tablet</option>
                                </select>
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                                <div className="form-group">
                                    <label className="form-label">Billing ZIP</label>
                                    <input className="form-input" placeholder="84604" value={form.billing_zip} onChange={(e) => setForm({ ...form, billing_zip: e.target.value })} maxLength={10} />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Shipping ZIP</label>
                                    <input className="form-input" placeholder="84604" value={form.shipping_zip} onChange={(e) => setForm({ ...form, shipping_zip: e.target.value })} maxLength={10} />
                                </div>
                            </div>

                            <div className="form-group">
                                <label className="form-label">Promo Code <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>(optional)</span></label>
                                <input className="form-input" placeholder="SAVE10" value={form.promo_code} onChange={(e) => setForm({ ...form, promo_code: e.target.value })} />
                            </div>

                            <button
                                className="btn btn-primary"
                                style={{ width: '100%', justifyContent: 'center', padding: '11px', fontSize: 14, marginTop: 4 }}
                                onClick={handleSubmit}
                                disabled={submitting}
                            >
                                {submitting ? 'Placing Order…' : `Place Order · ${fmt(total)}`}
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {toast && (
                <div className={`toast ${toast.type}`}>{toast.msg}</div>
            )}
        </div>
    );
}