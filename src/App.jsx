import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Shield, Camera, Upload, FileText, AlertTriangle, Loader2, Play, CheckCircle, Trash2, Pause, RotateCcw, Smartphone, Square } from 'lucide-react';

// Swap this string with your live Render app URL once deployed
const BACKEND_URL = "http://localhost:8000";

const App = () => {
  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState("idle"); 
  const [isLive, setIsLive] = useState(false);
  const [connectedTopics, setConnectedTopics] = useState([]);
  const [ntfyTopicInput, setNtfyTopicInput] = useState("");
  const [showPopup, setShowPopup] = useState(false);
  const [showRtspModal, setShowRtspModal] = useState(false);
  const [rtspInput, setRtspInput] = useState("");

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    const handleUnload = () => {
      connectedTopics.forEach(topic => {
        fetch(`${BACKEND_URL}/unsubscribe_ntfy`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ topic: topic }),
          keepalive: true 
        });
      });
      fetch(`${BACKEND_URL}/clear_logs`, { method: "DELETE", keepalive: true });
    };
    window.addEventListener("beforeunload", handleUnload);
    return () => window.removeEventListener("beforeunload", handleUnload);
  }, [connectedTopics]);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const logRes = await axios.get(`${BACKEND_URL}/get_latest_alerts`);
        setLogs(logRes.data.alerts);
      } catch (e) {
        console.error("Sync error");
      }
    }, 1500); 
    return () => clearInterval(interval);
  }, []);

  // FRONTEND EXECUTION: Starts the webcam processing loop locally inside the user's browser
  const startWebcam = async () => {
    setIsLive(true);
    setStatus("playing");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 640, height: 480 }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      // Initialize your client-side model loop here
      requestAnimationFrame(processCanvasFrame);
    } catch (err) {
      alert("Webcam permission denied or unavailable.");
      setIsLive(false);
      setStatus("idle");
    }
  };

  // CLIENT SIDE INFERENCE SIMULATION: Runs client-side bounding box operations
  const processCanvasFrame = () => {
    if (status !== "playing" && !isLive) return;
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx && videoRef.current) {
      ctx.drawImage(videoRef.current, 0, 0, 640, 480);
      
      // --- VIVA NOTE: Place your loaded onnxruntime execution script here ---
      // Inside this local loop, when an anomaly is verified, fire it directly to the cloud backend:
      // axios.post(`${BACKEND_URL}/register_incident`, { event_type: "Fire", timestamp: "12:00:00", timestamp_val: Date.now() / 1000 });
    }
    if (isLive) requestAnimationFrame(processCanvasFrame);
  };

  const handleStopStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    setIsLive(false);
    setStatus("idle");
  };

  const handleClearLogs = async () => {
    await axios.delete(`${BACKEND_URL}/clear_logs`);
    setLogs([]);
  };

  const handleDownloadReport = async () => {
    window.open(`${BACKEND_URL}/download_final_report`, '_blank');
  };

  const handleSubscribe = async () => {
    const newTopic = ntfyTopicInput.trim();
    if (!newTopic || connectedTopics.includes(newTopic)) return;
    try {
      await axios.post(`${BACKEND_URL}/subscribe_ntfy`, { topic: newTopic });
      setConnectedTopics([...connectedTopics, newTopic]);
      setNtfyTopicInput("");
    } catch (e) {
      alert("Backend connection error.");
    }
  };

  const handleRemoveTopic = async (topicToRemove) => {
    await axios.post(`${BACKEND_URL}/unsubscribe_ntfy`, { topic: topicToRemove });
    setConnectedTopics(connectedTopics.filter(t => t !== topicToRemove));
  };

  return (
    <div className="dashboard-container">
      {/* Dynamic Responsive Stylesheet injection */}
      <style>{`
        .dashboard-container { display: flex; min-height: 100vh; width: 100vw; backgroundColor: #080a0f; color: #e0e0e0; fontFamily: sans-serif; overflow-x: hidden; }
        .sidebar { width: 280px; backgroundColor: #0f141d; padding: 30px; display: flex; flexDirection: column; gap: 15px; borderRight: '1px solid #1e2533'; }
        .main-content { flex: 1; padding: 20px; display: flex; flexDirection: column; gap: 20px; }
        .grid-layout { display: grid; gridTemplateColumns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; width: 100%; }
        .video-box { backgroundColor: #000; borderRadius: 8px; border: 1px solid #1e2533; minHeight: 350px; display: flex; justify-content: center; align-items: center; position: relative; }
        .control-panel { backgroundColor: #0f141d; padding: 15px; borderRadius: 8px; display: flex; items-center; gap: 15px; }
        .panel-card { backgroundColor: #0f141d; borderRadius: 8px; border: 1px solid #1e2533; display: flex; flexDirection: column; height: 100%; }
        @media (max-width: 1023px) {
          .dashboard-container { flexDirection: column; }
          .sidebar { width: 100%; box-sizing: border-box; }
        }
      `}</style>

      <div className="sidebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '20px', fontWeight: 'bold', color: '#fff' }}>
          <Shield size={32} color="#3d5afe" /> <span>AI GUARD</span>
        </div>
        <button style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '12px', backgroundColor: '#1a202c', color: '#fff', border: '1px solid #2d3748', borderRadius: '6px', cursor: 'pointer' }} onClick={() => setShowRtspModal(true)}>
          <Camera size={20}/> Connect Live Stream
        </button>
        <button style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '12px', backgroundColor: '#3d5afe', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold', justifyContent: 'center' }} onClick={startWebcam}>
          <Play size={20}/> Connect Local Camera
        </button>
        <div style={{ marginTop: 'auto' }}>
          <button style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '12px', backgroundColor: 'transparent', color: '#a0aec0', border: '1px solid #2d3748', borderRadius: '6px', cursor: 'pointer', width: '100%' }} onClick={handleDownloadReport}>
            <FileText size={20}/> Download Final Report
          </button>
        </div>
      </div>

      <div className="main-content">
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: '#a0aec0' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: isLive ? '#00e676' : '#ff1744', boxShadow: '0 0 10px currentcolor' }} />
            SYSTEM {status.toUpperCase()}
          </div>
          <h1 style={{ fontSize: '22px', fontWeight: '300', margin: 0 }}>Surveillance Command Center</h1>
        </header>

        <div className="grid-layout">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            <div className="video-box">
              <video ref={videoRef} autoPlay playsInline muted style={{ display: 'none' }} />
              {!isLive ? (
                <div style={{ textAlign: 'center', color: '#4a5568' }}>
                  <Play size={48} /> <p>Connect a camera to start hardware-accelerated tracking.</p>
                </div>
              ) : (
                <canvas ref={canvasRef} width="640" height="480" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              )}
            </div>

            {isLive && (
              <div className="control-panel">
                <button onClick={handleStopStream} style={{ backgroundColor: '#ff1744', border: 'none', color: '#fff', padding: '10px', borderRadius: '50%', cursor: 'pointer' }}>
                  <Square size={16} fill="#fff" />
                </button>
                <span style={{ color: '#00e676', fontSize: '12px', fontWeight: 'bold' }}>EDGE RUNTIME ACTIVE (CLIENT-SIDE)</span>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            <div className="panel-card" style={{ flex: 1 }}>
              <div style={{ padding: '15px', fontSize: '12px', fontWeight: 'bold', borderBottom: '1px solid #1e2533', color: '#3d5afe', display: 'flex', justifyContent: 'space-between' }}>
                <span>FORENSIC INCIDENT LOGS</span>
                <button onClick={handleClearLogs} style={{ background: 'none', border: 'none', color: '#718096', cursor: 'pointer' }}><Trash2 size={16} /></button>
              </div>
              <div style={{ padding: '15px', overflowY: 'auto', maxH: '250px' }}>
                {logs.length === 0 ? (
                  <p style={{ color: '#4a5568', fontSize: '12px', textAlign: 'center' }}>No threats detected in current session.</p>
                ) : (
                  logs.map((log, index) => (
                    <div key={index} style={{ display: 'flex', gap: '12px', padding: '10px', backgroundColor: '#161d29', borderRadius: '6px', marginBottom: '8px', borderLeft: '3px solid #ff1744' }}>
                      <AlertTriangle size={18} color="#ff1744" />
                      <div>
                        <div style={{ fontSize: '13px', fontWeight: '600', color: '#fff' }}>{log.event_type}</div>
                        <div style={{ fontSize: '11px', color: '#718096' }}>{log.timestamp}</div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="panel-card">
              <div style={{ padding: '15px', fontSize: '12px', fontWeight: 'bold', borderBottom: '1px solid #1e2533', color: '#3d5afe' }}>GET NOTIFIED</div>
              <div style={{ padding: '15px' }}>
                <button onClick={() => setShowPopup(true)} style={{ width: '100%', backgroundColor: '#3d5afe', color: '#fff', border: 'none', padding: '12px', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer' }}>
                  Manage Devices ({connectedTopics.length})
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Popups and Modals */}
      {showPopup && (
        <div style={{ position: 'fixed', top:0, left:0, right:0, bottom:0, backgroundColor: 'rgba(0,0,0,0.7)', display:'flex', alignItems:'center', justifyContent:'center', zIndex:1000 }}>
          <div style={{ backgroundColor: '#0f141d', padding: '30px', borderRadius: '8px', border: '1px solid #1e2533', width: '400px' }}>
            <h2 style={{ color: '#3d5afe', margin: '0 0 15px 0' }}>Connected Devices</h2>
            {connectedTopics.map((topic, idx) => (
              <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', backgroundColor: '#161d29', padding: '10px', borderRadius: '6px', marginBottom: '8px' }}>
                <span style={{ color: '#00e676' }}>{topic}</span>
                <button onClick={() => handleRemoveTopic(topic)} style={{ background: 'none', border: 'none', color: '#ff1744', cursor: 'pointer' }}><Trash2 size={16}/></button>
              </div>
            ))}
            <div style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
              <input type="text" placeholder="Topic name..." value={ntfyTopicInput} onChange={(e) => setNtfyTopicInput(e.target.value)} style={{ flex: 1, padding: '10px', backgroundColor: '#161d29', border: '1px solid #2d3748', borderRadius: '6px', color: '#fff' }} />
              <button onClick={handleSubscribe} style={{ backgroundColor: '#3d5afe', border: 'none', padding: '10px', color: '#fff', borderRadius: '6px' }}>Add</button>
            </div>
            <button onClick={() => setShowPopup(false)} style={{ width: '100%', background: 'transparent', border: '1px solid #2d3748', color: '#a0aec0', padding: '10px', borderRadius: '6px', marginTop: '15px', cursor: 'pointer' }}>Close</button>
          </div>
        </div>
      )}

      {showRtspModal && (
        <div style={{ position: 'fixed', top:0, left:0, right:0, bottom:0, backgroundColor: 'rgba(0,0,0,0.7)', display:'flex', alignItems:'center', justifyContent:'center', zIndex:1000 }}>
          <div style={{ backgroundColor: '#0f141d', padding: '30px', borderRadius: '8px', border: '1px solid #1e2533', width: '400px' }}>
            <h2 style={{ color: '#3d5afe', margin: '0 0 15px 0' }}>Connect RTSP Stream</h2>
            <input type="text" placeholder="rtsp://your-camera-ip..." value={rtspInput} onChange={(e) => setRtspInput(e.target.value)} style={{ width: '100%', padding: '12px', backgroundColor: '#161d29', border: '1px solid #2d3748', borderRadius: '6px', color: '#fff', boxSizing: 'border-box' }} />
            <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
              <button onClick={handleRtspSubmit} style={{ flex: 1, backgroundColor: '#3d5afe', border: 'none', padding: '12px', color: '#fff', borderRadius: '6px', fontWeight: 'bold' }}>Connect</button>
              <button onClick={() => setShowRtspModal(false)} style={{ flex: 1, background: 'transparent', border: '1px solid #2d3748', color: '#a0aec0', padding: '12px', borderRadius: '6px' }}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;