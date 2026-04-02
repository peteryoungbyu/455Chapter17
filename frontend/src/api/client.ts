const BASE_URL = 'http://localhost:5000/api';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
    const res = await fetch(`${BASE_URL}${path}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
}

// ── Types ──────────────────────────────────────────────────────────────────

export interface Customer {
    customer_id: number;
    full_name: string;
    email: string;
    city: string;
    state: string;
    customer_segment: string;
    loyalty_tier: string;
    is_active: boolean;
}

export interface Product {
    product_id: number;
    sku: string;
    product_name: string;
    category: string;
    price: number;
    is_active: boolean;
}

export interface Order {
    order_id: number;
    customer_id: number;
    customer_name?: string;
    order_datetime: string;
    payment_method: string;
    device_type: string;
    order_total: number;
    risk_score: number;
    is_fraud: boolean;
}

export interface PlaceOrderPayload {
    customer_id: number;
    payment_method: string;
    device_type: string;
    billing_zip: string;
    shipping_zip: string;
    promo_code?: string;
    items: { product_id: number; quantity: number }[];
}

export interface ScoredOrder {
    order_id: number;
    customer_name: string;
    order_total: number;
    risk_score: number;
    fraud_probability: number;
    scored_at: string;
}

// ── Endpoints ──────────────────────────────────────────────────────────────

export const api = {
    ping: () => request<{ message: string; timestampUtc: string }>('/fraud/ping'),

    getCustomers: () => request<Customer[]>('/customers'),
    getCustomer: (id: number) => request<Customer>(`/customers/${id}`),

    getProducts: () => request<Product[]>('/products'),

    placeOrder: (payload: PlaceOrderPayload) =>
        request<Order>('/orders', { method: 'POST', body: JSON.stringify(payload) }),

    getOrders: () => request<Order[]>('/orders'),

    runScoring: () =>
        request<{ message: string; scored: number }>('/scoring/run', { method: 'POST' }),

    getPriorityQueue: () => request<ScoredOrder[]>('/scoring/queue'),
};