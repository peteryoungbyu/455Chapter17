import { NavLink } from 'react-router-dom';

export default function Navbar() {
    return (
        <nav className="navbar">
            <NavLink to="/" className="navbar-brand">
                <span className="brand-dot" />
                ShieldCart
            </NavLink>
            <div className="navbar-links">
                <NavLink
                    to="/"
                    end
                    className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
                >
                    Customers
                </NavLink>
                <NavLink
                    to="/admin"
                    className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
                >
                    Admin
                </NavLink>
            </div>
        </nav>
    );
}