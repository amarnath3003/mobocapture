# MoboCapture

MoboCapture is a proposed fully open system for collecting human demonstrations that can become useful robot-training data. It is intended to run across a capability spectrum:

- an open reference head-mounted device;
- LiDAR-capable iPhones;
- ordinary iOS and Android phones;
- research headsets and third-party rigs through adapters;
- instrumented grippers, gloves, wrist cameras, and robots as optional higher-fidelity sources.

The initial research concludes that the project should **not** begin by building another isolated camera app or head-mounted recorder. Several 2026 projects already do portions of that. The defensible contribution is an open, device-independent data contract; calibrated raw capture; reproducible enrichment; explicit uncertainty and provenance; privacy tooling; and robot-policy validation across device tiers.

The complete landscape study, proposed architecture, schema, hardware direction, evaluation design, and phased roadmap are in [RESEARCH.md](RESEARCH.md).

The agreed first build stage is documented in [RGB_FIRST_MVP.md](RGB_FIRST_MVP.md): RGB-only hands/fingers, visual tracks, interaction hypotheses, and evidence-linked VLM descriptions before metric geometry or hardware work.

## Proposed one-line mission

> Capture human interaction anywhere, preserve what the sensors actually measured, derive robot-relevant representations reproducibly, and prove their value on real robots.

## Current status

The RGB-first foundation is now functional for six user-selectable modules. The CLI and React UI preserve and hash one input video, index original timestamps, compute deterministic quality signals, resolve dependencies, record model/license provenance, render a synchronized review overlay, and package the complete session.

- **Hands & Fingers:** pinned MediaPipe, persistent IDs, handedness, and 21 landmarks.
- **Objects:** pinned Apache-2.0 Grounding DINO and SAM2, user prompts, boxes, masks, and persistent IDs.
- **Motion Tracking:** persistent Shi-Tomasi/Lucas-Kanade points plus full-resolution dense Farneback flow arrays.
- **Hand-object interactions:** evidence-linked assignments, contact likelihood, grasp/release candidates, and explicit hypothesis status.
- **Privacy & Redaction:** YuNet faces plus grounded screens/documents/plates/mirrors, review rows, and a separate redacted MP4 while retaining the governed original.
- **Video Quality:** timing, decode, blur/exposure, duplicates, camera-motion class, and review overlay.

VLM descriptions are deliberately deferred for now. Experimental RGB geometry remains planned. The software reports planned processors as unavailable instead of fabricating outputs.

No claim is made yet that MoboCapture data improves a robot policy; producing that evidence remains the project's first major gate.

## Run the current foundation

Requirements: Python 3.10 or newer and FFmpeg/FFprobe on `PATH`.

```powershell
python -m pip install -e .
python -m mobocapture modules
python -m mobocapture process .\example.mp4 --output .\example.mobocapture
python -m mobocapture process .\hands.mp4 --module hands_fingers --output .\hands.mobocapture
python -m mobocapture process .\scene.mp4 --module objects --object-concept cup --object-concept tool --output .\scene.mobocapture
python -m mobocapture process .\task.mp4 --module hand_object_interactions --module privacy_redaction --object-concept cup --object-concept tool --output .\task.mobocapture
```

For the local upload interface:

```powershell
mobocapture web
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). Upload one video, select capability checkboxes, monitor the local job, preview the finished overlay, and download the complete session as a ZIP. Web sessions are stored in `mobocapture-web-data/` by default; use `--data-dir` to choose another location.

The browser interface is a React application built with Vite and served by the local FastAPI backend. To rebuild it after editing `frontend/`:

```powershell
cd frontend
npm install
npm run build
```

The default `foundation` profile enables the currently functional ingest, video-quality, and overlay processors. Module dependencies can be inspected without running models:

```powershell
python -m mobocapture resolve --module hand_object_interactions
python -m mobocapture resolve --profile rgb_core
```

Each run creates an immutable copied input under `raw/`, requested and resolved manifests under `manifests/`, per-processor run records, Parquet tables under `derived/v0.1/`, and `review/overlay.mp4`. Learned modules add their own tables: `hands.parquet`, `regions.parquet`, `point_tracks.parquet`, `optical_flow.parquet` plus compressed dense arrays, `interactions.parquet`, and `privacy_regions.parquet`. Privacy runs also add `review/redacted.mp4` and `manifests/redaction.json`. Per-frame companion tables include frames with zero detections. The overlay is a constant-frame-rate review artifact; `frame_index.parquet` remains canonical timing.

The hand-module validation downloads Google MediaPipe's pinned public sample, runs the complete video pipeline, asserts two persistent tracks with 21 landmarks each, and creates an original/output comparison:

```powershell
python scripts/validate_hands.py --output .\validation-output\hands
```

The object-module validator uses a hash-pinned public SAM2 fixture and checks Grounding DINO boxes, SAM2 masks, stable IDs, complete dataset rows, and the rendered comparison:

```powershell
python scripts/validate_objects.py --output .\validation-output\objects
```

The remaining non-generative module validators use hash-pinned public samples:

```powershell
python scripts/validate_motion.py --output .\validation-output\motion
python scripts/validate_interactions.py --output .\validation-output\interactions
python scripts/validate_privacy.py --output .\validation-output\privacy
```

The first object run downloads about 850 MB of pinned weights. Later runs reuse the local cache. CUDA is used when available; CPU execution is supported but substantially slower.

## Performance behavior

The pipeline keeps full frame coverage and the existing model/algorithm quality settings. It does not accelerate by dropping frames, reducing source resolution, changing thresholds, using a smaller model, or switching to reduced-precision inference.

On CUDA, full-resolution frames are evaluated in conservative batches of two using FP32. Object and privacy prompts remain separate, but reuse the exact Grounding DINO visual-backbone features for the same frames. Independent CPU processors run concurrently with the serialized GPU processors, which avoids GPU memory overcommit on 8 GB cards. Mask RLE generation is vectorized, dense-flow NPZ files use fast lossless compression, and already-compressed dataset files are stored directly in the final ZIP rather than recompressed.

Every processor run manifest records `wall_time_seconds`; learned processor manifests also record the inference batch size and precision. Batch size can be overridden through the programmatic processing options under `performance.gpu_batch_size`, while `performance.processor_workers` controls the independent processor worker limit. The defaults are tuned for the validated RTX 3070 8 GB environment.
