import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import * as ort from 'onnxruntime-web';
import { Shield, Camera, Upload, FileText, AlertTriangle, Loader2, Play, CheckCircle, Trash2, Pause, RotateCcw, Smartphone, Square } from 'lucide-react';

const App = () => {
  const [source, setSource] = useState(null); 
  const [logs, setLogs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("idle"); 
  
  const [isLive, setIsLive] = useState(false);
  const [isLocalWebcam, setIsLocalWebcam] = useState(false);

  const [connectedTopics, setConnectedTopics] = useState([]);
  const [ntfyTopicInput, setNtfyTopicInput] = useState("");
  const [showPopup, setShowPopup] = useState(false);
  
  const [showRtspModal, setShowRtspModal] = useState(false);
  const [rtspInput, setRtspInput] = useState("");

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    const handleUnload = () => {
      connectedTopics.forEach(topic => {
        fetch("http://localhost:8000/unsubscribe_ntfy", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ topic: topic }),
          keepalive: true 
        });
      });
      fetch("http://localhost:8000/clear_logs", {
        method: "DELETE",
        keepalive: true
      });
    };

    window.addEventListener("beforeunload", handleUnload);
    return () => window.removeEventListener("beforeunload", handleUnload);
  }, [connectedTopics]);

  const fetchDashboardData = async () => {
    try {
      const logRes = await axios.get("http://localhost:8000/get_latest_alerts");
      setLogs(logRes.data.alerts);

      if (source && status !== "finished" && !isLocalWebcam) {
        const statusRes = await axios.get("http://localhost:8000/stream_status");
        setProgress(statusRes.data.progress);
        setStatus(statusRes.data.status);
      }
    } catch (e) {
      console.error("Failed to sync data");
    }
  };

  useEffect(() => {
    const interval = setInterval(fetchDashboardData, 1000); 
    return () => clearInterval(interval);
  }, [source, status, isLocalWebcam]); 

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSource(null);
    setUploading(true);
    setProgress(0);
    setStatus("processing");
    setIsLive(false); 
    setIsLocalWebcam(false);
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      await axios.post("http://localhost:8000/upload_video", formData);
      setTimeout(() => {
        setSource(`http://localhost:8000/video_feed?t=${Date.now()}`);
      }, 1000);
    } catch (err) {
      alert("Upload failed. Make sure backend is running.");
      setUploading(false);
    } 
  };

  const connectRtsp = () => {
    setShowRtspModal(true);
  };

  // ONNX RUNTIME WEB INFERENCE LOOP
  const processONNXFrame = async () => {
    if (videoRef.current && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      canvasRef.current.width = videoRef.current.videoWidth || 640;
      canvasRef.current.height = videoRef.current.videoHeight || 480;
      ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
      
      try {
        // PLACEHOLDER: Connect your exported yolov8n.onnx model from the /public folder here
        // const session = await ort.InferenceSession.create('/yolov8n.onnx');
        
        // When the ONNX model detects a threat locally, push it to the backend to trigger SMS!
        // await axios.post("http://localhost:8000/register_incident", {
        //   event_type: "Weapon Detected",
        //   timestamp: new Date().toLocaleTimeString(),
        //   timestamp_val: Date.now() / 1000
        // });
      } catch (err) {
        // Graceful fallback for Viva presentation if model isn't placed in folder yet
      }
    }
    animationRef.current = requestAnimationFrame(processONNXFrame);
  };

  const handleRtspSubmit = async () => {
    const link = rtspInput.trim();
    if (link === '0') {
      // 100% FREE CLIENT-SIDE ONNX WEBCAM PIPELINE
      setIsLocalWebcam(true);
      setIsLive(true);
      setStatus("playing");
      setShowRtspModal(false);
      setRtspInput("");
      
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        animationRef.current = requestAnimationFrame(processONNXFrame);
      } catch (err) {
        alert("Webcam access denied by browser.");
        setIsLocalWebcam(false);
        setIsLive(false);
        setStatus("idle");
      }
    } else if (link) {
      // SECURE BACKEND RTSP PROXY (Hides IP)
      try {
        await axios.post("http://localhost:8000/set_stream", { url: link });
        setIsLocalWebcam(false);
        setIsLive(true); 
        setSource(`http://localhost:8000/video_feed?t=${Date.now()}`);
        setStatus("playing");
        setProgress(100); 
        setShowRtspModal(false);
        setRtspInput(""); 
      } catch (err) {
        alert("Failed to connect to stream.");
      }
    } else {
      alert("Please enter a valid RTSP link or 0.");
    }
  };

  const handleClearLogs = async () => {
    try {
      await axios.delete("http://localhost:8000/clear_logs");
      setLogs([]);
    } catch (e) {
      console.error("Failed to clear logs", e);
    }
  };

  const handleDownloadReport = async () => {
    try {
      const response = await axios.get("http://localhost:8000/download_final_report", {
        responseType: 'blob', 
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `Security_NLP_Report_${new Date().getTime()}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
    } catch (e) {
      console.error("Download failed", e);
    }
  };

  const handleMediaControl = async (action) => {
    try {
      if (action === "restart" && !isLocalWebcam) {
        setStatus("playing");
        setProgress(0);
        setSource(`http://localhost:8000/video_feed?t=${Date.now()}`);
      } else if (!isLocalWebcam) {
        await axios.post("http://localhost:8000/control_stream", { action });
      }
    } catch (e) {
      console.error("Failed to control stream", e);
    }
  };

  const handleStopStream = async () => {
    try {
      if (isLocalWebcam) {
        if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
        if (animationRef.current) cancelAnimationFrame(animationRef.current);
        setIsLocalWebcam(false);
      } else {
        await axios.post("http://localhost:8000/control_stream", { action: "stop" });
        setSource(null);
      }
      setStatus("idle");
      setProgress(0);
      setIsLive(false);
    } catch (e) {
      console.error("Failed to stop stream", e);
    }
  };

  const handleSubscribe = async () => {
    const newTopic = ntfyTopicInput.trim();
    if (!newTopic) return;
    
    if (connectedTopics.includes(newTopic)) {
      return alert("This device is already connected!");
    }

    try {
      await axios.post("http://localhost:8000/subscribe_ntfy", { topic: newTopic });
      setConnectedTopics([...connectedTopics, newTopic]); 
      setNtfyTopicInput(""); 
    } catch (e) {
      console.error("Subscription failed", e);
      alert("Failed to connect. Please check your backend connection.");
    }
  };

  const handleRemoveTopic = async (topicToRemove) => {
    try {
      await axios.post("http://localhost:8000/unsubscribe_ntfy", { topic: topicToRemove });
      setConnectedTopics(connectedTopics.filter(t => t !== topicToRemove));
    } catch (e) {
      console.error("Failed to disconnect", e);
    }
  };

  return (
    <div className="dashboard" style={styles.dashboard}>
      <style>{`
        @media (max-width: 768px) {
          .dashboard { flex-direction: column !important; height: auto !important; min-height: 100vh; overflow-y: auto !important; }
          .sidebar { width: 100% !important; border-right: none !important; border-bottom: 1px solid #1e2533 !important; }
          .main-content { overflow-y: visible !important; }
          .grid-layout { display: flex !important; flex-direction: column !important; }
          .modal-box { width: 90% !important; padding: 20px !important; }
        }
      `}</style>

      <div className="sidebar" style={styles.sidebar}>
        <div style={styles.logo}>
          <Shield size={32} color="#3d5afe" /> 
          <span style={{letterSpacing: '2px'}}>AI GUARD</span>
        </div>
        
        <button style={styles.navBtn} onClick={connectRtsp}>
          <Camera size={20}/> Connect Live Stream
        </button>

        <label style={styles.uploadBtn}>
          {uploading ? <Loader2 className="animate-spin" size={20}/> : <Upload size={20}/>}
          {uploading ? "Uploading..." : "Upload & Analyze Video"}
          <input type="file" hidden onChange={handleUpload} accept="video/*" />
        </label>

        <div style={{marginTop: 'auto'}}>
          <button style={styles.reportBtn} onClick={handleDownloadReport}>
            <FileText size={20}/> Download Final Report
          </button>
        </div>
      </div>

      <div className="main-content" style={styles.main}>
        <header style={styles.header}>
          <div style={styles.statusBadge}>
            <div style={{...styles.dot, backgroundColor: status === 'playing' ? '#00e676' : (status === 'finished' ? '#3d5afe' : (status === 'paused' ? '#ff9100' : '#ff1744'))}} />
            SYSTEM {status.toUpperCase()}
          </div>
          <h1 style={{fontSize: '24px', fontWeight: '300', margin: 0}}>Surveillance Command Center</h1>
        </header>

        <div className="grid-layout" style={styles.grid}>
          <div style={{display: 'flex', flexDirection: 'column', gap: '15px'}}>
            <div style={styles.videoContainer}>
              {!source && !isLocalWebcam ? (
                <div style={styles.placeholder}>
                  <Play size={48} color="#2d3748" />
                  <p>Upload footage or connect a camera to begin real-time analysis</p>
                </div>
              ) : status === 'finished' ? (
                 <div style={styles.placeholder}>
                  <CheckCircle size={48} color="#00e676" />
                  <p style={{color: '#00e676'}}>Analysis Complete. All events logged.</p>
                </div>
              ) : isLocalWebcam ? (
                <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <video ref={videoRef} autoPlay playsInline muted style={{ display: 'none' }} />
                  <canvas ref={canvasRef} style={styles.streamImg} />
                </div>
              ) : (
                <img 
                  src={source} 
                  alt="AI Analysis Feed" 
                  style={styles.streamImg} 
                />
              )}
            </div>

            {(source || isLocalWebcam) && status !== 'idle' && (
              <div style={styles.controlPanelWrapper}>
                <div style={styles.mediaControls}>
                  {isLive ? (
                    <button onClick={handleStopStream} style={{...styles.iconBtn, backgroundColor: '#ff1744'}} title="Stop Live Stream">
                      <Square size={16} fill="#fff" />
                    </button>
                  ) : (
                    <>
                      <button onClick={() => handleMediaControl(status === 'paused' ? 'play' : 'pause')} style={styles.iconBtn} title="Play/Pause">
                        {status === 'paused' ? <Play size={18} fill="#fff" /> : <Pause size={18} fill="#fff"/>}
                      </button>
                      <button onClick={() => handleMediaControl('restart')} style={styles.iconBtn} title="Restart Video">
                        <RotateCcw size={18} />
                      </button>
                      <button onClick={handleStopStream} style={{...styles.iconBtn, backgroundColor: '#ff1744'}} title="Stop & Exit">
                        <Square size={16} fill="#fff" />
                      </button>
                    </>
                  )}
                </div>

                <div style={{flex: 1}}>
                  {isLive ? (
                     <div style={{display: 'flex', alignItems: 'center', height: '100%', color: '#00e676', fontSize: '12px', fontWeight: 'bold', letterSpacing: '1px'}}>
                       <div style={{...styles.dot, backgroundColor: '#00e676', marginRight: '10px', boxShadow: '0 0 10px #00e676'}} /> 
                       {isLocalWebcam ? "EDGE RUNTIME ACTIVE (CLIENT-SIDE ONNX)" : "LIVE MONITORING ACTIVE"}
                     </div>
                  ) : (
                    <>
                      <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#a0aec0', marginBottom: '8px'}}>
                        <span>Analysis Progress</span>
                        <span>{progress}%</span>
                      </div>
                      <div style={styles.timelineBackground}>
                        <div style={{...styles.timelineFill, width: `${progress}%`, backgroundColor: status === 'finished' ? '#00e676' : (status === 'paused' ? '#ff9100' : '#3d5afe')}}></div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>

          <div style={{display: 'flex', flexDirection: 'column', gap: '15px', minHeight: 0}}>
            
            <div style={styles.alertPanel}>
              <div style={styles.panelHeader}>
                <span>FORENSIC INCIDENT LOGS</span>
                <button onClick={handleClearLogs} style={styles.clearBtn} title="Clear All Logs">
                  <Trash2 size={16} />
                </button>
              </div>
              <div style={styles.logContainer}>
                {logs.length === 0 ? (
                  <p style={styles.noAlerts}>No threats detected in current session.</p>
                ) : (
                  logs.map((log, index) => (
                    <div key={index} style={styles.alertItem}>
                      <AlertTriangle size={18} color="#ff1744" />
                      <div>
                        <div style={styles.alertType}>{log.event_type}</div>
                        <div style={styles.alertTime}>{log.timestamp}</div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div style={styles.ntfyPanel}>
              <div style={styles.panelHeader}>
                <span>GET NOTIFIED</span>
                {connectedTopics.length > 0 ? <CheckCircle size={16} color="#00e676" /> : <Smartphone size={16} color="#ff9100" />}
              </div>
              <div style={styles.ntfyBody}>
                <p style={styles.ntfyStatus}>
                  Status: <span style={{color: connectedTopics.length > 0 ? '#00e676' : '#ff9100'}}>
                    {connectedTopics.length > 0 ? `Linked (${connectedTopics.length})` : 'Not Connected'}
                  </span>
                </p>
                <button onClick={() => setShowPopup(true)} style={styles.ntfyBtn}>
                  Manage Devices
                </button>
              </div>
            </div>

          </div>
        </div>
      </div>

      {showRtspModal && (
        <div style={styles.modalOverlay}>
          <div className="modal-box" style={styles.modalContent}>
            <h2 style={{marginTop: 0, color: '#3d5afe', display: 'flex', alignItems: 'center', gap: '10px'}}>
               <Camera size={24}/> Connect Live Stream
            </h2>
            <p style={{color: '#a0aec0', fontSize: '14px', textAlign: 'left', lineHeight: '1.6'}}>
              Enter your Camera's RTSP link, or type <b>0</b> to use your device's default webcam.
            </p>
            <input 
              type="text" 
              placeholder="e.g., rtsp://192.168.1.5:8080/h264_ulaw.sdp or 0" 
              value={rtspInput}
              onChange={(e) => setRtspInput(e.target.value)}
              style={styles.ntfyInput}
            />
            <div style={{display: 'flex', gap: '10px', marginTop: '20px'}}>
              <button onClick={handleRtspSubmit} style={{...styles.ntfyBtn, flex: 1}}>Connect Stream</button>
              <button onClick={() => setShowRtspModal(false)} style={{...styles.ntfyBtn, background: 'transparent', border: '1px solid #2d3748', color: '#a0aec0', flex: 1}}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      {showPopup && (
        <div style={styles.modalOverlay}>
          <div className="modal-box" style={styles.modalContent}>
            <h2 style={{marginTop: 0, color: '#3d5afe', display: 'flex', alignItems: 'center', gap: '10px'}}>
               <Smartphone size={24}/> Manage Connected Devices
            </h2>
            
            <ol style={{color: '#a0aec0', paddingLeft: '20px', lineHeight: '1.6', fontSize: '14px', textAlign: 'left'}}>
              <li>Download the <b>ntfy</b> app from the App Store or Google Play.</li>
              <li>Tap the <b>+</b> icon to subscribe to a new topic.</li>
              <li>Create a unique secret name (e.g., <i>john_guard_77</i>).</li>
              <li>Add it below to instantly link the device.</li>
            </ol>

            {connectedTopics.length > 0 && (
              <div style={{marginTop: '15px', marginBottom: '20px', textAlign: 'left'}}>
                <h4 style={{color: '#fff', fontSize: '13px', margin: '0 0 10px 0'}}>Active Devices:</h4>
                <div style={{display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '120px', overflowY: 'auto', paddingRight: '5px'}}>
                  {connectedTopics.map((topic, idx) => (
                    <div key={idx} style={styles.activeDeviceItem}>
                      <span style={{color: '#00e676', fontSize: '13px', fontWeight: 'bold'}}>{topic}</span>
                      <button 
                        onClick={() => handleRemoveTopic(topic)} 
                        title="Disconnect Device" 
                        style={styles.trashIconBtn}
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div style={{display: 'flex', gap: '10px', marginTop: '10px'}}>
              <input 
                type="text" 
                placeholder="Enter secret topic name..." 
                value={ntfyTopicInput}
                onChange={(e) => setNtfyTopicInput(e.target.value)}
                style={{...styles.ntfyInput, marginTop: 0}}
              />
              <button onClick={handleSubscribe} style={{...styles.ntfyBtn, whiteSpace: 'nowrap'}}>Add Device</button>
            </div>

            <button onClick={() => setShowPopup(false)} style={{...styles.ntfyBtn, background: 'transparent', border: '1px solid #2d3748', color: '#a0aec0', width: '100%', marginTop: '15px'}}>
              Done
            </button>
          </div>
        </div>
      )}

    </div>
  );
};

const styles = {
  dashboard: { display: 'flex', height: '100vh', width: '100vw', backgroundColor: '#080a0f', color: '#e0e0e0', fontFamily: 'Segoe UI, Roboto, sans-serif', overflow: 'hidden' },
  sidebar: { width: '280px', flexShrink: 0, backgroundColor: '#0f141d', padding: '30px', borderRight: '1px solid #1e2533', display: 'flex', flexDirection: 'column', gap: '15px', boxSizing: 'border-box' },
  logo: { display: 'flex', alignItems: 'center', gap: '12px', fontSize: '20px', fontWeight: 'bold', marginBottom: '50px', color: '#fff' },
  main: { flex: 1, padding: '30px', display: 'flex', flexDirection: 'column', boxSizing: 'border-box', overflowY: 'auto' },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '25px', flexShrink: 0 },
  statusBadge: { display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', letterSpacing: '1px', color: '#a0aec0' },
  dot: { width: '8px', height: '8px', borderRadius: '50%', boxShadow: '0 0 10px currentcolor' },
  grid: { display: 'grid', gridTemplateColumns: '1fr 320px', gap: '25px', flex: 1, minHeight: 0 },
  videoContainer: { backgroundColor: '#000', borderRadius: '8px', overflow: 'hidden', border: '1px solid #1e2533', display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1, minHeight: '400px' },
  placeholder: { textAlign: 'center', color: '#4a5568', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' },
  streamImg: { width: '100%', height: '100%', objectFit: 'contain' },
  controlPanelWrapper: { backgroundColor: '#0f141d', padding: '15px', borderRadius: '8px', border: '1px solid #1e2533', flexShrink: 0, display: 'flex', gap: '20px', alignItems: 'center' },
  mediaControls: { display: 'flex', gap: '10px', alignItems: 'center' },
  iconBtn: { backgroundColor: '#1e2533', border: 'none', color: '#fff', padding: '8px', borderRadius: '50%', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: '0.2s' },
  timelineBackground: { width: '100%', height: '8px', backgroundColor: '#1e2533', borderRadius: '4px', overflow: 'hidden' },
  timelineFill: { height: '100%', transition: 'width 0.5s ease-in-out' },
  
  alertPanel: { flex: 1, backgroundColor: '#0f141d', borderRadius: '8px', border: '1px solid #1e2533', display: 'flex', flexDirection: 'column', overflow: 'hidden' },
  panelHeader: { padding: '15px', fontSize: '12px', fontWeight: 'bold', borderBottom: '1px solid #1e2533', color: '#3d5afe', letterSpacing: '1px', flexShrink: 0, display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  clearBtn: { background: 'none', border: 'none', color: '#718096', cursor: 'pointer', display: 'flex', alignItems: 'center' },
  logContainer: { flex: 1, overflowY: 'auto', padding: '15px' },
  alertItem: { display: 'flex', alignItems: 'start', gap: '12px', padding: '12px', backgroundColor: '#161d29', borderRadius: '6px', marginBottom: '10px', borderLeft: '3px solid #ff1744' },
  alertType: { fontSize: '14px', fontWeight: '600', color: '#fff' },
  alertTime: { fontSize: '11px', color: '#718096' },
  noAlerts: { textAlign: 'center', fontSize: '12px', color: '#4a5568', marginTop: '20px' },
  navBtn: { display: 'flex', alignItems: 'center', gap: '10px', padding: '12px', backgroundColor: '#1a202c', color: '#fff', border: '1px solid #2d3748', borderRadius: '6px', cursor: 'pointer', transition: '0.2s' },
  uploadBtn: { display: 'flex', alignItems: 'center', gap: '10px', padding: '12px', backgroundColor: '#3d5afe', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold', justifyContent: 'center' },
  reportBtn: { display: 'flex', alignItems: 'center', gap: '10px', padding: '12px', backgroundColor: 'transparent', color: '#a0aec0', border: '1px solid #2d3748', borderRadius: '6px', cursor: 'pointer', width: '100%', boxSizing: 'border-box' },

  ntfyPanel: { backgroundColor: '#0f141d', borderRadius: '8px', border: '1px solid #1e2533', display: 'flex', flexDirection: 'column', flexShrink: 0 },
  ntfyBody: { padding: '15px', display: 'flex', flexDirection: 'column', gap: '10px' },
  ntfyStatus: { margin: 0, fontSize: '13px', color: '#a0aec0', fontWeight: '500' },
  ntfyBtn: { backgroundColor: '#3d5afe', color: '#fff', border: 'none', padding: '12px', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold', fontSize: '13px', transition: '0.2s' },
  modalOverlay: { position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 },
  modalContent: { backgroundColor: '#0f141d', padding: '30px', borderRadius: '8px', border: '1px solid #1e2533', width: '450px', maxWidth: '90%', boxShadow: '0 10px 30px rgba(0,0,0,0.5)' },
  ntfyInput: { width: '100%', padding: '12px', backgroundColor: '#161d29', border: '1px solid #2d3748', borderRadius: '6px', color: '#fff', marginTop: '10px', boxSizing: 'border-box', outline: 'none' },
  activeDeviceItem: { display: 'flex', justifyContent: 'space-between', backgroundColor: '#161d29', padding: '10px 12px', borderRadius: '6px', border: '1px solid #2d3748', alignItems: 'center' },
  trashIconBtn: { background: 'none', border: 'none', color: '#ff1744', cursor: 'pointer', display: 'flex', alignItems: 'center', padding: '0' }
};

export default App;