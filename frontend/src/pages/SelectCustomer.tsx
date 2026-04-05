import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { Customer } from '../api/client';

// ── helpers ──────────────────────────────────────────────────────────────

function initials(name: string) {
  return name
    .split(' ')
    .map((w) => w[0])
    .slice(0, 2)
    .join('')
    .toUpperCase();
}

function tierBadge(tier: string) {
  const t = (tier ?? '').toLowerCase();
  if (t === 'gold') return 'badge badge-gold';
  if (t === 'silver') return 'badge badge-silver';
  if (t === 'bronze') return 'badge badge-bronze';
  return 'badge badge-standard';
}

const AVATAR_COLORS = [
  '#00c8ff22',
  '#006eff22',
  '#7c3aed22',
  '#00e5a022',
  '#f5a62322',
];

function avatarColor(id: number) {
  return AVATAR_COLORS[id % AVATAR_COLORS.length];
}

// ── mock fallback ─────────────────────────────────────────────────────────

const MOCK_CUSTOMERS: Customer[] = [
  {
    customer_id: 1,
    full_name: 'Alice Nguyen',
    email: 'alice@example.com',
    city: 'Austin',
    state: 'TX',
    customer_segment: 'Premium',
    loyalty_tier: 'Gold',
    is_active: true,
  },
  {
    customer_id: 2,
    full_name: 'Brian Okafor',
    email: 'brian@example.com',
    city: 'Chicago',
    state: 'IL',
    customer_segment: 'Standard',
    loyalty_tier: 'Silver',
    is_active: true,
  },
  {
    customer_id: 3,
    full_name: 'Clara Mendez',
    email: 'clara@example.com',
    city: 'Miami',
    state: 'FL',
    customer_segment: 'Premium',
    loyalty_tier: 'Gold',
    is_active: true,
  },
  {
    customer_id: 4,
    full_name: 'David Kim',
    email: 'dkim@example.com',
    city: 'Seattle',
    state: 'WA',
    customer_segment: 'Standard',
    loyalty_tier: 'Bronze',
    is_active: true,
  },
  {
    customer_id: 5,
    full_name: 'Elena Russo',
    email: 'elena@example.com',
    city: 'Denver',
    state: 'CO',
    customer_segment: 'Premium',
    loyalty_tier: 'Silver',
    is_active: true,
  },
  {
    customer_id: 6,
    full_name: 'Frank Osei',
    email: 'fosei@example.com',
    city: 'Atlanta',
    state: 'GA',
    customer_segment: 'Standard',
    loyalty_tier: 'Standard',
    is_active: true,
  },
];

// ── component ─────────────────────────────────────────────────────────────

export default function SelectCustomer() {
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    api
      .getCustomers()
      .then(setCustomers)
      .catch(() => setCustomers(MOCK_CUSTOMERS))
      .finally(() => setLoading(false));
  }, []);

  const filtered = customers.filter((c) =>
    `${c.full_name} ${c.email} ${c.city}`
      .toLowerCase()
      .includes(search.toLowerCase())
  );

  return (
    <div className="page-shell">
      <div className="page-content">
        <div
          className="page-header fade-in"
          style={{
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 16,
          }}
        >
          <div>
            <h1>Select Customer</h1>
            <p>Choose an account to place an order or view history.</p>
          </div>
          <input
            className="form-input"
            style={{ width: 260 }}
            placeholder="Search by name, email, city…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        {loading ? (
          <div className="spinner" />
        ) : filtered.length === 0 ? (
          <div className="empty-state">
            <p>No customers found.</p>
          </div>
        ) : (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
              gap: 16,
            }}
          >
            {filtered.map((c, i) => (
              <button
                key={c.customer_id}
                className="stagger-child"
                onClick={() => navigate(`/customer/${c.customer_id}`)}
                style={{
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius)',
                  padding: '20px 22px',
                  textAlign: 'left',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 16,
                  cursor: 'pointer',
                  transition: 'all 0.18s ease',
                  animationDelay: `${i * 0.05}s`,
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.borderColor =
                    'var(--border-bright)';
                  (e.currentTarget as HTMLElement).style.transform =
                    'translateY(-2px)';
                  (e.currentTarget as HTMLElement).style.boxShadow =
                    '0 8px 24px rgba(0,0,0,0.3)';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.borderColor =
                    'var(--border)';
                  (e.currentTarget as HTMLElement).style.transform = '';
                  (e.currentTarget as HTMLElement).style.boxShadow = '';
                }}
              >
                <div
                  style={{
                    width: 48,
                    height: 48,
                    borderRadius: '50%',
                    background: avatarColor(c.customer_id),
                    border: '1px solid var(--border-bright)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontFamily: 'var(--font-display)',
                    fontWeight: 700,
                    fontSize: 16,
                    color: 'var(--accent)',
                    flexShrink: 0,
                  }}
                >
                  {initials(c.full_name)}
                </div>

                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      marginBottom: 3,
                    }}
                  >
                    <span
                      style={{
                        fontFamily: 'var(--font-display)',
                        fontWeight: 600,
                        fontSize: 15,
                        color: 'var(--text)',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {c.full_name}
                    </span>
                    <span className={tierBadge(c.loyalty_tier)}>
                      {c.loyalty_tier}
                    </span>
                  </div>
                  <div
                    style={{
                      fontSize: 12.5,
                      color: 'var(--text-secondary)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {c.email}
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      color: 'var(--text-muted)',
                      marginTop: 2,
                    }}
                  >
                    {c.city}, {c.state} · {c.customer_segment}
                  </div>
                </div>

                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 16 16"
                  fill="none"
                  style={{ color: 'var(--text-muted)', flexShrink: 0 }}
                >
                  <path
                    d="M6 3l5 5-5 5"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
