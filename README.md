# Image and Video Tools for Pipeline Creator

A collection of image and video visualization, processing, playback, and recording modules for [**Pipeline Creator**](https://github.com/n3odym3/Pipeline_creator).

This repository contains image and video modules (in-memory image sequence container with batch ROI analysis, high-performance image viewer with ROI inspection, ultra-low-latency video viewer, recursive video file reader with batch queues, and synchronized multi-backend video writer) designed for plug-and-play integration into the Pipeline Creator ecosystem.

---

## Installation

To install and use one or more modules in **Pipeline Creator**:

1. **Install Module Dependencies**:

   Install the required Python packages and tools for the module(s) you intend to use within your Pipeline Creator environment:
   ```bash
   pip install opencv-python numpy dearpygui loguru vidgear
   # Optional: ensure FFmpeg is installed and accessible on PATH for hardware-accelerated video encoding (NVENC, QuickSync, AMF)
   ```

2. **Copy Modules to Pipeline Creator**:
   
   Copy and paste the folder(s) of the desired module(s) (such as `Image_container`, `Image_viewer`, `Video_reader`, `Video_viewer`, `Video_writer`, etc.) directly into the `modules/` folder of your **Pipeline Creator** installation:
   ```
   Pipeline_creator/
   ├── modules/
   │   ├── Image_container/
   │   ├── Image_viewer/
   │   ├── Video_reader/
   │   ├── Video_viewer/
   │   ├── Video_writer/
   │   └── ...
   ```

3. **Launch Pipeline Creator**:

   The copied modules will be automatically detected and made available in the Pipeline Creator node editor.

---

## Available Modules

| Module | Description | Key Dependencies |
| :--- | :--- | :--- |
| [**Image_container**](./Image_container/) | In-memory image sequence manager, caching engine, and batch playback module. Supports directory loading or live frame ingestion, biological group and substring filtering, asynchronous ROI batch mean intensity analysis across entire image sequences, experiment persistence (compressed ZIP import/export with metadata), and sequential downstream frame distribution. | `opencv-python`, `numpy`, `dearpygui`, `loguru` |
| [**Image_viewer**](./Image_viewer/) | High-performance image visualization and analysis module with real-time GPU texture rendering. Features interactive Region of Interest (ROI) inspection with live statistics and crop streaming, dynamic colormapping (`B&W`, `Inferno`, `Jet`, `HSV`), 8/12/16-bit intensity rescaling and autoremap, transparent alpha mask blending, and automated export/clipboard tools. | `opencv-python`, `numpy`, `dearpygui`, `loguru` |
| [**Video_reader**](./Video_reader/) | Hierarchical video file reader and batch queue navigator. Scans directories and nested subfolders recursively with interactive tree views, provides full transport controls (Play, Pause, Stop, scrubbing slider, adjustable FPS), state color-coding (active vs completed), format filtering, and automated sequential batch queue processing (`Process ALL`) with downstream completion trigger synchronization. | `opencv-python`, `numpy`, `dearpygui`, `loguru` |
| [**Video_viewer**](./Video_viewer/) | High-performance, ultra-low-latency (600+ FPS) video streaming display module. Decouples frame ingestion and rendering via a dedicated background worker thread with latest-frame dropping, dual FPS telemetry (Read vs Display), automatic color / mask overlay conversion, and configurable GPU texture modes (`raw` buffer copy or `static`). | `opencv-python`, `numpy`, `dearpygui`, `loguru` |
| [**Video_writer**](./Video_writer/) | High-performance video recording module with smart real-time frame rate synchronization (Drop & Duplicate). Supports multi-backend encoding (VidGear `WriteGear`, direct FFmpeg subprocess pipe with GPU acceleration like NVENC / QuickSync / AMF, OpenCV fallback), exotic formats (Mono 8-bit, 12-bit / 16-bit HDR, BGRA), SIMD tone mapping, and live recording telemetry. | `opencv-python`, `numpy`, `dearpygui`, `vidgear`, `loguru`, FFmpeg (optional/recommended) |

---
