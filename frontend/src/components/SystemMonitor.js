import React, { useState, useEffect } from 'react';
import './SystemMonitor.css';

function SystemMonitor() {
    const [containers, setContainers] = useState({});
    const [gpuStatus, setGpuStatus] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                setIsLoading(true);
                
                // Fetch container status
                const containersResponse = await fetch('/api/monitor/containers');
                const containersData = await containersResponse.json();
                setContainers(containersData);

                // Fetch GPU status
                const gpuResponse = await fetch('/api/monitor/gpu');
                const gpuData = await gpuResponse.json();
                setGpuStatus(gpuData);
                
            } catch (error) {
                console.error('Error fetching monitoring data:', error);
            } finally {
                setIsLoading(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="system-monitor">
            <h2>System Monitor</h2>
            
            {/* GPU Status */}
            <div className="monitor-section">
                <h3>GPU Status</h3>
                {isLoading && (
                    <p className="loading-indicator">Fetching latest data...</p>
                )}
                {gpuStatus && !gpuStatus.error ? (
                    <div className="gpu-card">
                        <div className="metric">
                            <label>GPU Utilization:</label>
                            <div className="progress-bar">
                                <div 
                                    className="progress" 
                                    style={{width: `${gpuStatus.utilization}%`}}
                                />
                                <span>{gpuStatus.utilization}%</span>
                            </div>
                        </div>
                        <div className="metric">
                            <label>GPU Memory:</label>
                            <div className="progress-bar">
                                <div 
                                    className="progress"
                                    style={{width: `${(gpuStatus.memory.used / gpuStatus.memory.total) * 100}%`}}
                                />
                                <span>{gpuStatus.memory.used}/{gpuStatus.memory.total} {gpuStatus.memory.unit}</span>
                            </div>
                        </div>
                    </div>
                ) : null}
            </div>

            {/* Container Status */}
            <div className="monitor-section">
                <h3>Container Status</h3>
                <div className="container-grid">
                    {Object.entries(containers).map(([name, status]) => (
                        <div key={name} className={`container-card ${status.running ? 'running' : 'stopped'}`}>
                            <h4>{name}</h4>
                            <p className="status">Status: {status.status}</p>
                            {status.memory_usage && (
                                <p>Memory: {Math.round(status.memory_usage.used / 1024 / 1024)} MB</p>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

export default SystemMonitor; 