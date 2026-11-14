import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './index.css';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          backgroundColor: '#0f172a',
          color: '#f87171',
          padding: '30px',
          fontFamily: 'monospace',
          border: '2px solid #ef4444',
          borderRadius: '12px',
          margin: '20px'
        }}>
          <h2 style={{ color: '#ef4444', marginTop: 0 }}>⚠️ React Rendering Exception</h2>
          <p><strong>Error:</strong> {this.state.error?.toString()}</p>
          <pre style={{
            background: '#1e293b',
            color: '#e2e8f0',
            padding: '16px',
            borderRadius: '8px',
            overflowX: 'auto',
            fontSize: '12px'
          }}>
            {this.state.error?.stack}
          </pre>
          {this.state.errorInfo && (
            <details style={{ marginTop: '16px' }}>
              <summary style={{ cursor: 'pointer', color: '#38bdf8' }}>Component Stack</summary>
              <pre style={{
                background: '#1e293b',
                color: '#94a3b8',
                padding: '16px',
                borderRadius: '8px',
                overflowX: 'auto',
                fontSize: '11px',
                marginTop: '8px'
              }}>
                {this.state.errorInfo.componentStack}
              </pre>
            </details>
          )}
        </div>
      );
    }
    return this.props.children;
  }
}

const rootEl = document.getElementById('root');
if (!rootEl) {
  console.error("Root element #root not found!");
} else {
  try {
    ReactDOM.createRoot(rootEl).render(
      <React.StrictMode>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </React.StrictMode>,
    );
  } catch (err) {
    console.error("Fatal exception during ReactDOM.createRoot render:", err);
    rootEl.innerHTML = `<div style="color:red;padding:20px;font-family:monospace">Failed to mount app: ${err.message}</div>`;
  }
}
