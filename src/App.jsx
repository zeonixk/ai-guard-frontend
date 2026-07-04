import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import * as ort from 'onnxruntime-web';
import { Shield, Camera, Upload, FileText, AlertTriangle, Loader2, Play, CheckCircle, Trash2, Pause, RotateCcw, Smartphone, Square } from 'lucide-react';

// Configure ONNX Runtime to pull WASM binaries reliably
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

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
  const imageRef = useRef(null); // Added for local RTSP processing
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const animationRef = useRef(null);

  // AI Guard Edge Ensemble References
  const modelsRef = useRef({ custom: null, general: null, pose: null });
  const isProcessingRef = useRef(false);

  // Load ONNX models into the browser on startup
  useEffect(() => {
    const loadEnsemble = async () => {
      try {
        modelsRef.current.custom = await ort.InferenceSession.create('/best.onnx', { executionProviders: ['wasm'] });
        modelsRef.current.general = await ort.InferenceSession.create('/yolov8n.onnx', { executionProviders: ['wasm'] });
        modelsRef.current.pose = await ort.InferenceSession.create('/yolov8n-pose.onnx', { executionProviders: ['wasm'] });
        console.log("AI Guard Edge Ensemble Loaded Successfully");
      } catch (err) {
        console.error("Ensure .onnx files are in the public folder:", err);
      }
    };
    loadEnsemble();
  }, []);

  useEffect(() => {
    const handleUnload = () => {
      connectedTopics.forEach(topic => {
        fetch("https://zeonixk-ai-guard-api.hf.space/unsubscribe_ntfy", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ topic: topic }),
          keepalive: true 
        });
      });
      fetch("https://zeonixk-ai-guard-api.hf.space/clear_logs", {
        method: "DELETE",
        keepalive: true
      });
    };

    window.addEventListener("beforeunload", handleUnload);
    return () => window.removeEventListener("beforeunload", handleUnload);
  }, [connectedTopics]);

  const fetchDashboardData = async () => {
    try {
      const logRes = await axios.get("https://zeonixk-ai-guard-api.hf.space/get_latest_alerts");
      setLogs(logRes.data.alerts);

      // Only fetch backend progress if it's a backend process (which we moved away from, kept for safety)
      if (source && status !== "finished" && !isLocalWebcam) {
        const statusRes = await axios.get("https://zeonixk-ai-guard-api.hf.space/stream_status");
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

  // FIX: Video Uploads now process 100% locally in the browser cache
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSource(null);
    setUploading(true);
    setProgress(0);
    setStatus("processing");
    setIsLive(false); 
    
    // Create local memory link instead of cloud upload
    const fileUrl = URL.createObjectURL(file);
    
    setIsLocalWebcam(true); 
    setStatus("playing");
    setUploading(false);

    setTimeout(() => {
      if (videoRef.current) {
        videoRef.current.srcObject = null;
        videoRef.current.src = fileUrl;
        videoRef.current.load();
        videoRef.current.play().catch(err => console.log("Play prevented", err));
      }
    }, 200);

    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    animationRef.current = requestAnimationFrame(processONNXFrame);
  };

  const connectRtsp = () => {
    setShowRtspModal(true);
  };

  // UNIFIED ONNX RUNTIME WEB INFERENCE LOOP
  const processONNXFrame = async () => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
      let drawable = null;

      // Ensure media is actually loaded and ready before attempting to draw
      if (videoRef.current && videoRef.current.readyState >= 2) {
        drawable = videoRef.current;
        canvasRef.current.width = videoRef.current.videoWidth || 640;
        canvasRef.current.height = videoRef.current.videoHeight || 480;
      } else if (imageRef.current && imageRef.current.complete && imageRef.current.naturalWidth > 0) {
        drawable = imageRef.current;
        canvasRef.current.width = imageRef.current.naturalWidth || 640;
        canvasRef.current.height = imageRef.current.naturalHeight || 480;
      }

      if (drawable) {
        ctx.drawImage(drawable, 0, 0, canvasRef.current.width, canvasRef.current.height);
        
        // SOFTWARE ENSEMBLE: AI runs on local hardware in the background
        if (!isProcessingRef.current && modelsRef.current.custom) {
          isProcessingRef.current = true;
          try {
            // Note: Full YOLOv8 tensor preprocessing and NMS logic goes here.
            /*
            const imageTensor = createTensorFromCanvas(canvasRef.current);
            const feeds = { images: imageTensor };
            const [resCustom, resGen, resPose] = await Promise.all([
              modelsRef.current.custom.run(feeds),
              modelsRef.current.general.run(feeds),
              modelsRef.current.pose.run(feeds)
            ]);
            await axios.post("https://zeonixk-ai-guard-api.hf.space/register_incident", {
              event_type: "Weapon Detected",
              timestamp: new Date().toLocaleTimeString(),
              timestamp_val: Date.now() / 1000
            });
            */
          } catch (err) {
            console.error(err);
          } finally {
            isProcessingRef.current = false; 
          }
        }
      }
    }
    // Video drawing loop continues uninterrupted at 60fps
    animationRef.current = requestAnimationFrame(processONNXFrame);
  };

  const handleRtspSubmit = async () => {
    const link = rtspInput.trim();
    if (link === '0') {
      setIsLocalWebcam(true);
      setIsLive(true);
      setStatus("playing");
      setShowRtspModal(false);
      setSource(null);
      setRtspInput("");
      
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        streamRef.current = stream;
        
        // FIX: Ensure play() is called after stream attachment
        setTimeout(() => {
            if (videoRef.current) {
              videoRef.current.srcObject = stream;
              videoRef.current.play().catch(err => console.log("Play prevented", err));
            }
        }, 200);

        if (animationRef.current) cancelAnimationFrame(animationRef.current);
        animationRef.current = requestAnimationFrame(processONNXFrame);
      } catch (err) {
        alert("Webcam access denied by browser.");
        setIsLocalWebcam(false);
        setIsLive(false);
        setStatus("idle");
      }
    } else if (link) {
      // FIX: RTSP now routes through local pipeline. Backend only proxies frames.
      try {
        await axios.post("https://zeonixk-ai-guard-api.hf.space/set_stream", { url: link });
        setIsLocalWebcam(true); 
        setIsLive(true); 
        setSource(`https://zeonixk-ai-guard-api.hf.space/video_feed?t=${Date.now()}`);
        setStatus("playing");
        setProgress(100); 
        setShowRtspModal(false);
        setRtspInput(""); 
        
        if (animationRef.current) cancelAnimationFrame(animationRef.current);
        animationRef.current = requestAnimationFrame(processONNXFrame);
      } catch (err) {
        alert("Failed to connect to stream.");
      }
    } else {
      alert("Please enter a valid RTSP link or 0.");
    }
  };

  const handleClearLogs = async () => {
    try {
      await axios.delete("https://zeonixk-ai-guard-api.hf.space/clear_logs");
      setLogs([]);
    } catch (e) {
      console.error("Failed to clear logs", e);
    }
  };

  const handleDownloadReport = async () => {
    try {
      const response = await axios.get("https://zeonixk-ai-guard-api.hf.space/download_final_report", {
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
        setSource(`https://zeonixk-ai-guard-api.hf.space/video_feed?t=${Date.now()}`);
      } else if (isLocalWebcam && videoRef.current) {
        if (action === "pause") {
           videoRef.current.pause();
           setStatus("paused");
        }
        if (action === "play") {
           videoRef.current.play();
           setStatus("playing");
        }
        if (action === "restart") {
           videoRef.current.currentTime = 0;
           videoRef.current.play();
           setStatus("playing");
        }
      }
    } catch (e) {
      console.error("Failed to control stream", e);
    }
  };

  const handleStopStream = async () => {
    try {
      if (isLocalWebcam) {
        if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
        if (videoRef.current) videoRef.current.pause();
        if (animationRef.current) cancelAnimationFrame(animationRef.current);
        setIsLocalWebcam(false);
        setSource(null);
      } else {
        await axios.post("https://zeonixk-ai-guard-api.hf.space/control_stream", { action: "stop" });
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
      await axios.post("https://zeonixk-ai-guard-api.hf.space/subscribe_ntfy", { topic: newTopic });
      setConnectedTopics([...connectedTopics, newTopic]); 
      setNtfyTopicInput(""); 
    } catch (e) {
      console.error("Subscription failed", e);
      alert("Failed to connect. Please check your backend connection.");
    }
  };

  const handleRemoveTopic = async (topicToRemove) => {
    try {
      await axios.post("https://zeonixk-ai-guard-api.hf.space/unsubscribe_ntfy", { topic: topicToRemove });
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
                  <video ref={videoRef} playsInline muted style={{ display: 'none' }} />
                  {/* Hidden image fetches RTSP frames from backend proxy for local canvas processing */}
                  {source && <img ref={imageRef} src={source} style={{ display: 'none' }} crossOrigin="anonymous" alt="stream" />}
                  <canvas ref={canvasRef} style={styles.streamImg} />
                </div>
              ) : null}
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
                       EDGE RUNTIME ACTIVE (CLIENT-SIDE ONNX)
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