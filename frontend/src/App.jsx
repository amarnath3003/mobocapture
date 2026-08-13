import { useEffect, useRef, useState } from "react";

const TERMINAL_JOB_STATES = new Set(["completed", "failed"]);

function ArrowIcon({ download = false }) {
  return download ? (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 4v12m0 0 4-4m-4 4-4-4M5 20h14" />
    </svg>
  ) : (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 12h14m-5-5 5 5-5 5" />
    </svg>
  );
}

function Header() {
  return (
    <header className="site-header">
      <a className="brand" href="/" aria-label="MoboCapture home">
        <span className="brand-mark" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
        <span>MoboCapture</span>
      </a>
      <div className="local-badge">
        <span /> Local processing
      </div>
    </header>
  );
}

function Hero() {
  return (
    <section className="hero">
      <p className="eyebrow">RGB-FIRST DATA FOUNDATION</p>
      <h1>
        Turn a video into a
        <br />
        <em>robot-learning dataset.</em>
      </h1>
      <p className="hero-copy">
        Upload one RGB recording, choose the evidence you need, and receive a traceable
        dataset plus a visual review overlay.
      </p>
    </section>
  );
}

function PanelHeading({ step, title, aside, id }) {
  return (
    <div className="panel-heading">
      <div>
        <span className="step">{step}</span>
        <h2 id={id}>{title}</h2>
      </div>
      {aside}
    </div>
  );
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes === 0) return "0 bytes";
  const units = ["bytes", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${value.toFixed(index ? 1 : 0)} ${units[index]}`;
}

function SourceVideoPanel({ file, onFile, onClear, onMetadata }) {
  const [dragging, setDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return undefined;
    }
    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  const acceptDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    const dropped = [...event.dataTransfer.files];
    onFile(dropped.find((candidate) => candidate.type.startsWith("video/")) || dropped[0]);
  };

  return (
    <section className="panel upload-panel" aria-labelledby="upload-title">
      <PanelHeading
        step="01"
        id="upload-title"
        title="Source video"
        aside={<span className="required">REQUIRED</span>}
      />

      {!file ? (
        <label
          id="drop-zone"
          className={`drop-zone${dragging ? " dragging" : ""}`}
          htmlFor="video-input"
          onDragEnter={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={acceptDrop}
        >
          <input
            id="video-input"
            name="video"
            type="file"
            accept="video/*,.mkv,.mov,.avi"
            required
            onChange={(event) => onFile(event.target.files?.[0])}
          />
          <span className="upload-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" />
            </svg>
          </span>
          <strong>Drop your video here</strong>
          <span>or click to browse from your computer</span>
          <small>MP4, MOV, MKV, WebM, AVI · processed on this machine</small>
        </label>
      ) : (
        <div id="file-card" className="file-card">
          <video
            id="source-preview"
            src={previewUrl}
            muted
            playsInline
            onLoadedMetadata={(event) => {
              const video = event.currentTarget;
              onMetadata({
                duration: video.duration,
                width: video.videoWidth,
                height: video.videoHeight,
              });
            }}
          />
          <div className="file-details">
            <strong id="file-name">{file.name}</strong>
            <span id="file-meta">
              {formatBytes(file.size)} · {file.type || "video"}
            </span>
          </div>
          <button
            id="remove-file"
            className="icon-button"
            type="button"
            aria-label="Remove video"
            onClick={onClear}
          >
            ×
          </button>
        </div>
      )}
    </section>
  );
}

function CapabilityPanel({
  catalog,
  selected,
  onToggle,
  onPreset,
  loadingError,
  objectConcepts,
  onObjectConcepts,
  videoMetadata,
}) {
  const denseSelected = selected.has("dense_optical_flow");
  const denseBytes = denseSelected && videoMetadata
    ? videoMetadata.width * videoMetadata.height * 4 * 30 * videoMetadata.duration * 0.9
    : 0;
  return (
    <section className="panel features-panel" aria-labelledby="features-title">
      <PanelHeading
        step="02"
        id="features-title"
        title="Select capabilities"
        aside={
          <span id="selection-count" className="selection-count">
            {selected.size} SELECTED
          </span>
        }
      />

      <div className="preset-row" aria-label="Capability presets">
        <button type="button" data-preset="foundation" onClick={() => onPreset("foundation")}>
          Foundation
        </button>
        <button type="button" data-preset="rgb_core" onClick={() => onPreset("rgb_core")}>
          RGB core
        </button>
        <button type="button" data-preset="full_rgb" onClick={() => onPreset("full_rgb")}>
          Everything RGB
        </button>
      </div>

      <div id="module-grid" className="module-grid" aria-live="polite">
        {loadingError ? (
          <div className="module-loading">{loadingError}</div>
        ) : catalog.modules.length === 0 ? (
          <div className="module-loading">Loading capability registry…</div>
        ) : (
          catalog.modules.map((module) => {
            const checked = selected.has(module.id);
            return (
              <label
                className={`module-option${checked ? " selected" : ""}`}
                key={module.id}
              >
                <input
                  type="checkbox"
                  value={module.id}
                  checked={checked}
                  onChange={() => onToggle(module.id)}
                />
                <span className="checkbox" aria-hidden="true" />
                <span className="module-title">
                  {module.label}
                  <span className={`status-pill ${module.status}`}>{module.status}</span>
                  {module.cost_tier && module.cost_tier !== "standard" && (
                    <span className={`cost-pill ${module.cost_tier}`}>{module.cost_tier}</span>
                  )}
                </span>
                <p>{module.description}</p>
                <small className="storage-note">{module.storage_note}</small>
                {module.warning && checked && (
                  <small className="module-warning">{module.warning}</small>
                )}
              </label>
            );
          })
        )}
      </div>
      {["objects", "hand_object_interactions", "vlm_descriptions"].some((id) => selected.has(id)) && (
        <label className="object-concepts" htmlFor="object-concepts">
          <span>Objects to find</span>
          <input
            id="object-concepts"
            type="text"
            value={objectConcepts}
            onChange={(event) => onObjectConcepts(event.target.value)}
            placeholder="cup, bottle, tool, drawer"
          />
          <small>Comma-separated names or phrases; the defaults cover common robot scenes.</small>
        </label>
      )}
      {denseSelected && (
        <div className="size-warning" role="alert">
          <strong>Large raster output selected</strong>
          <span>
            {denseBytes > 0
              ? `Estimated dense-flow payload: roughly ${formatBytes(denseBytes)} for this video.`
              : "At 1080p30, dense flow is roughly 13 GB for each minute."}
          </span>
          <span>Use Sparse motion tracks for the normal robot-learning dataset.</span>
        </div>
      )}
      <p className="module-note">
        <span>i</span> Planned processors remain selectable and will be listed as unavailable
        until their validated model adapters are installed.
      </p>
    </section>
  );
}

function ProcessingPanel({ job }) {
  const progress = job?.progress ?? 0;
  return (
    <section id="progress-panel" className="panel progress-panel" aria-live="polite">
      <div className="processing-heading">
        <div className="spinner" aria-hidden="true" />
        <div>
          <p className="eyebrow">PROCESSING SESSION</p>
          <h2 id="progress-stage">{job?.stage || "Uploading video"}</h2>
        </div>
        <strong id="progress-value">{progress}%</strong>
      </div>
      <div className="progress-track">
        <span id="progress-bar" style={{ width: `${progress}%` }} />
      </div>
      <p id="progress-detail">
        {job?.status === "queued"
          ? "Waiting for the local processing worker."
          : `Processing ${job?.source_name || "your video"} on this machine.`}
      </p>
    </section>
  );
}

function ResultPanel({ job, onReset }) {
  const unavailable = job.unavailable_processors || [];
  return (
    <section id="result-panel" className="result-section" aria-live="polite">
      <div className="result-heading">
        <div>
          <p className="eyebrow">SESSION COMPLETE</p>
          <h2>Your dataset is ready.</h2>
        </div>
        <a id="download-link" className="primary-button" href={job.download_url}>
          <span>Download dataset</span>
          <ArrowIcon download />
        </a>
        {job.redacted_url && (
          <a className="secondary-button" href={job.redacted_url}>
            Download redacted video
          </a>
        )}
      </div>

      <div className="result-grid">
        <div className="overlay-card">
          <div className="video-shell">
            <video id="overlay-video" src={job.overlay_url} controls playsInline />
          </div>
          <div className="overlay-caption">
            <span /> Review overlay · canonical timing remains in frame_index.parquet
          </div>
        </div>
        <aside className="summary-card">
          <h3>Session summary</h3>
          <dl>
            <div>
              <dt>Status</dt>
              <dd id="summary-status">{job.session_status}</dd>
            </div>
            <div>
              <dt>Session ID</dt>
              <dd id="summary-id">{job.session_id}</dd>
            </div>
            <div>
              <dt>Completed</dt>
              <dd id="summary-completed">
                {job.completed_processors?.join(", ") || "None"}
              </dd>
            </div>
          </dl>
          {unavailable.length > 0 && (
            <div id="unavailable-box" className="unavailable-box">
              <strong>Planned processors</strong>
              <p id="summary-unavailable">{unavailable.join(", ")}</p>
            </div>
          )}
          <button id="new-session" className="secondary-button" type="button" onClick={onReset}>
            Process another video
          </button>
        </aside>
      </div>
    </section>
  );
}

function ErrorPanel({ message, onDismiss }) {
  return (
    <section id="error-panel" className="error-panel" role="alert">
      <div>
        <strong>Processing failed</strong>
        <p id="error-message">{message}</p>
      </div>
      <button id="dismiss-error" type="button" onClick={onDismiss}>
        Try again
      </button>
    </section>
  );
}

function App() {
  const [catalog, setCatalog] = useState({ modules: [], profiles: {} });
  const [catalogError, setCatalogError] = useState("");
  const [selected, setSelected] = useState(new Set());
  const [objectConcepts, setObjectConcepts] = useState(
    "person, cup, bottle, bowl, plate, box, bag, phone, tool, scissors, cloth, drawer, door",
  );
  const [file, setFile] = useState(null);
  const [videoMetadata, setVideoMetadata] = useState(null);
  const [phase, setPhase] = useState("form");
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const pollToken = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    async function loadCatalog() {
      try {
        const response = await fetch("/api/modules", { signal: controller.signal });
        if (!response.ok) throw new Error("Could not load the capability registry");
        const payload = await response.json();
        setCatalog(payload);
        setSelected(new Set(payload.profiles.foundation || ["video_quality"]));
      } catch (requestError) {
        if (requestError.name !== "AbortError") setCatalogError(requestError.message);
      }
    }
    loadCatalog();
    return () => controller.abort();
  }, []);

  useEffect(
    () => () => {
      pollToken.current += 1;
    },
    [],
  );

  const toggleModule = (moduleId) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(moduleId)) next.delete(moduleId);
      else next.add(moduleId);
      return next;
    });
  };

  const applyPreset = (profile) => {
    setSelected(new Set(catalog.profiles[profile] || []));
  };

  const pollJob = async (jobId, token) => {
    while (pollToken.current === token) {
      const response = await fetch(`/api/jobs/${jobId}`);
      if (!response.ok) throw new Error("Could not read processing status");
      const nextJob = await response.json();
      setJob(nextJob);
      if (TERMINAL_JOB_STATES.has(nextJob.status)) {
        if (nextJob.status === "completed") setPhase("result");
        else {
          setError(nextJob.error || "Unknown processing error");
          setPhase("error");
        }
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 700));
    }
  };

  const submitJob = async (event) => {
    event.preventDefault();
    if (!file || selected.size === 0) return;
    const token = pollToken.current + 1;
    pollToken.current = token;
    setPhase("processing");
    setError("");
    setJob({ progress: 0, stage: "Uploading video", source_name: file.name });

    const body = new FormData();
    body.append("video", file, file.name);
    selected.forEach((moduleId) => body.append("modules", moduleId));
    if (
      ["objects", "hand_object_interactions", "vlm_descriptions"].some((id) => selected.has(id)) &&
      objectConcepts.trim()
    ) {
      body.append("object_concepts", objectConcepts.trim());
    }

    try {
      const response = await fetch("/api/jobs", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Upload failed");
      setJob(payload);
      await pollJob(payload.job_id, token);
    } catch (requestError) {
      if (pollToken.current === token) {
        setError(requestError.message);
        setPhase("error");
      }
    }
  };

  const reset = () => {
    pollToken.current += 1;
    setFile(null);
    setJob(null);
    setError("");
    setPhase("form");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <>
      <Header />
      <main>
        <Hero />

        {(phase === "form" || phase === "error") && (
          <form id="job-form" className="workspace" noValidate onSubmit={submitJob}>
            <SourceVideoPanel
              file={file}
              onFile={(nextFile) => {
                setFile(nextFile);
                setVideoMetadata(null);
              }}
              onClear={() => {
                setFile(null);
                setVideoMetadata(null);
              }}
              onMetadata={setVideoMetadata}
            />
            <CapabilityPanel
              catalog={catalog}
              selected={selected}
              onToggle={toggleModule}
              onPreset={applyPreset}
              loadingError={catalogError}
              objectConcepts={objectConcepts}
              onObjectConcepts={setObjectConcepts}
              videoMetadata={videoMetadata}
            />
            <div className="action-row">
              <p>
                <strong>Nothing leaves this machine.</strong>
                <br />
                The original is copied and hash-verified.
              </p>
              <button
                id="process-button"
                className="primary-button"
                type="submit"
                disabled={!file || selected.size === 0}
              >
                <span>Build dataset</span>
                <ArrowIcon />
              </button>
            </div>
          </form>
        )}

        {phase === "processing" && <ProcessingPanel job={job} />}
        {phase === "result" && <ResultPanel job={job} onReset={reset} />}
        {phase === "error" && <ErrorPanel message={error} onDismiss={() => setPhase("form")} />}
      </main>
      <footer>
        <span>MoboCapture v0.1 · React</span>
        <span>Measured ≠ estimated ≠ hypothesized</span>
      </footer>
    </>
  );
}

export default App;
