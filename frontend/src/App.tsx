import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import SelectCustomer from './pages/SelectCustomer';
import PlaceOrder from './pages/PlaceOrder';
import AdminDashboard from './pages/AdminDashboard';
import './index.css';

export default function App() {
  return (
      <BrowserRouter>
        <Navbar />
        <Routes>
          <Route path="/" element={<SelectCustomer />} />
          <Route path="/customer/:id" element={<PlaceOrder />} />
          <Route path="/admin" element={<AdminDashboard />} />
        </Routes>
      </BrowserRouter>
  );
}