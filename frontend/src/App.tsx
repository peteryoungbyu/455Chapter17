import { useState } from 'react';
import './App.css';

function App() {
  const [result, setResult] = useState('No request made yet.');
  const [isLoading, setIsLoading] = useState(false);

  const testApiConnection = async () => {
    setIsLoading(true);
    setResult('Calling backend...');

    try {
      const response = await fetch('http://localhost:5000/api/fraud/ping');

      if (!response.ok) {
        setResult(`Request failed with status ${response.status}`);
        return;
      }

      const data = await response.json();
      setResult(`Success: ${data.message} at ${data.timestampUtc}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      setResult(`Connection error: ${message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="page">
      <section className="card">
        <h1>Frontend to Backend Connectivity Test</h1>
        <p>
          Frontend: <strong>http://localhost:3000</strong>
        </p>
        <p>
          Backend: <strong>http://localhost:5000</strong>
        </p>
        <button onClick={testApiConnection} disabled={isLoading}>
          {isLoading ? 'Testing...' : 'Test API Call'}
        </button>
        <pre>{result}</pre>
      </section>
    </main>
  );
}

export default App;
