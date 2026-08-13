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

Research and architecture definition only. No claim is made yet that MoboCapture data improves a robot policy; producing that evidence is the project's first major gate.
