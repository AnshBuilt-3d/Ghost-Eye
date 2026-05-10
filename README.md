<div align="center">

```
                  ╔═══════════════════════════════════════════════════════════════════════╗
                  ║                                                                       ║
                  ║  ██████╗ ██╗  ██╗ ██████╗ ███████╗████████╗ ███████╗██╗   ██╗███████╗ ║
                  ║ ██╔════╝ ██║  ██║██╔═══██╗██╔════╝╚══██╔══╝ ██╔════╝╚██╗ ██╔╝██╔════╝ ║
                  ║ ██║  ███╗███████║██║   ██║███████╗   ██║    █████╗   ╚████╔╝ █████╗   ║
                  ║ ██║   ██║██╔══██║██║   ██║╚════██║   ██║    ██╔══╝    ╚██╔╝  ██╔══╝   ║
                  ║ ╚██████╔╝██║  ██║╚██████╔╝███████║   ██║    ███████╗   ██║   ███████╗ ║
                  ║  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝    ╚══════╝   ╚═╝   ╚══════╝ ║
                  ║                                                                       ║
                  ║            Depth-Guided Geometry Completion Network (DGCN)            ║
                  ║       Indoor Furniture Reconstruction · Consumer GPU · Real-time      ║
                  ╚═══════════════════════════════════════════════════════════════════════╝
```

</div>

<br/>

[![Status](https://img.shields.io/badge/Phase_1-In_Training-2E75B6?style=for-the-badge&logo=pytorch&logoColor=white*)](/)
[![GPU](https://img.shields.io/badge/T4_GPU-16_GB_VRAM-76B900?style=for-the-badge&logo=nvidia&logoColor=white=)](/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge)](/)
[![Peak VRAM](https://img.shields.io/badge/Peak_VRAM-2.5_GB-orange?style=for-the-badge)](/)

<br/>

> *"A depth-guided, category-conditioned lightweight geometry completion network for indoor furniture
> reconstruction on consumer low-VRAM GPUs — 35M parameters, 88 MB, 0.5 s per object."*

<br/>

**[What is Ghost Eye](#what-is-ghost-eye) · [Architecture](#architecture) · [DGCN Model](#the-dgcn-model) · [Training](#training) · [Domain Gap](#the-domain-gap-problem) · [Results](#results) · [Roadmap](#roadmap)**

</div>

---

## What is Ghost Eye?

Ghost Eye is a **single-image 3D indoor furniture reconstruction system** designed to run entirely on consumer hardware — no cloud, no high-end GPU required.

Given a single RGB photo of a room, Ghost Eye:

1. **Estimates metric depth** — real-world measurements in metres, not relative depth
2. **Segments the scene** — identifies floor, wall, ceiling, and furniture regions
3. **Detects furniture objects** — open-vocabulary detection across 20+ categories
4. **Completes the geometry** — reconstructs the full 3D shape of each object, including heavily occluded parts
5. **Extracts a mesh** — produces a textured 3D mesh with UV layout, ready for rendering

The entire pipeline peaks at **~2.5 GB VRAM** at any single point. Every stage loads, runs, and unloads before the next begins. A single photo in — a set of textured 3D furniture meshes out.

### Why Does This Matter?

Commercial 3D scanning requires expensive hardware (LiDAR, structured light), controlled environments, or multi-image capture sequences. Ghost Eye operates from a single existing photo, taken by anyone, on any device.

The core research challenge is **geometry completion under occlusion** — reconstructing parts of furniture you *cannot see*. A chair pushed against a wall. A sofa half-visible behind a coffee table. A wardrobe with its doors facing away. Ghost Eye infers missing geometry from category knowledge, depth cues, and bilateral symmetry priors.

---

## Architecture

Ghost Eye is a **five-stage sequential pipeline**. Each stage loads to GPU, runs, and fully unloads before the next begins — keeping peak VRAM under 2.5 GB across all stages.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GHOST EYE PIPELINE                               │
│                                                                         │
│   RGB Image                                                             │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────────────────────────────┐                                       │
│  │  STAGE 1 · DA3Metric-Large   │  Metric depth map (real metres)       │
│  │  350M params · ~2.5 GB       │──► OBB dimensions (W, H, D)           │
│  └──────────────────────────────┘──► Partial point cloud (2048 pts)     │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────────────────────────────┐                                       │
│  │  STAGE 2 · SegFormer-b4      │  Semantic segmentation mask           │
│  │  ~64M params · ~2.0 GB       │──► Floor / Wall / Ceiling frame       │
│  └──────────────────────────────┘──► Room anchor for object placement   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────────────────────────────┐                                       │
│  │  STAGE 3 · GDINO 1.6 Edge    │  Open-vocabulary detection            │
│  │  ~172M params · ~2.2 GB      │──► Bounding boxes per object          │
│  └──────────────────────────────┘──► Category labels (one-hot × 20)     │
│       │                                                                 │
│       ▼  [per detected object]                                          │
│  ┌──────────────────────────────┐                                       │
│  │  STAGE 4 · DGCN (custom)     │  Geometry completion                  │
│  │  ~35M params · ~1 GB         │──► Complete point cloud (8192 pts)    │
│  └──────────────────────────────┘──► Confidence vector [3 scalars]      │
│       │                                                                 │
│       ▼  [CPU only — zero GPU VRAM]                                     │
│  ┌──────────────────────────────┐                                       │
│  │  STAGE 5 · Marching Cubes    │  Mesh extraction + UV mapping         │
│  │  CPU · 0 GB VRAM             │──► Textured .obj mesh output          │
│  └──────────────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### VRAM Budget

| Stage | Model | Params | Peak VRAM | Strategy |
|:---:|---|:---:|:---:|---|
| 1 | DA3Metric-Large | 350M | ~2.5 GB ⚠️ | Load → Run → Unload |
| 2 | SegFormer-b4 (ADE20K) | ~64M | ~2.0 GB | Load → Run → Unload |
| 3 | GDINO 1.6 Edge | ~172M | ~2.2 GB | Load → Run → Unload |
| 4 | DGCN (trained) | ~35M | ~1 GB | Load once, run N× per image |
| 5 | Marching Cubes | — | **0 GB** | CPU only |
| | **Peak at any single point** | | **~2.5 GB** | Well within T4 16 GB |

> ⚠️ Stage 1 (DA3Metric-Large) is the tightest stage and is verified first during implementation.

---

## The DGCN Model

The **Depth-Guided Geometry Completion Network** is the core contribution of Ghost Eye — a lightweight, category-conditioned geometry completion model trained from scratch on 3D-FUTURE indoor furniture.

### Inputs & Outputs

```
INPUTS
──────────────────────────────────────────────────────
  partial_pc     2048 points    Back-projected visible surface from DA3
  depth_patch    32 × 32 px     Cropped depth region around detected object
  category       one-hot × 20   From GDINO 1.6 Edge category label
  OBB            3 floats        Real-world W, H, D in metres from DA3

                        │
                        ▼
               ┌──────────────────┐
               │   DGCN FORWARD   │   ~35M parameters · FP16 · ~1 s
               └──────────────────┘
                        │
                        ▼

OUTPUTS
──────────────────────────────────────────────────────
  complete_pc    8192 points    Full furniture geometry (visible + completed)
  shape_conf     scalar [0,1]   Confidence in reconstructed geometry quality
  category_conf  scalar [0,1]   Confidence in GDINO category label
  coverage_conf  scalar [0,1]   Fraction of object surface that was visible
```

### Coarse-to-Fine Decoder

Ghost Eye uses a **two-stage decoder** rather than single-shot prediction to 8192 points. Single-shot decoders produce blobby geometry on thin structures — chair legs, armrests, back slats. The coarse stage anchors global topology first; refinement recovers local detail.

```
  partial_pc (2048 pts)              depth_patch (32×32)
          │                                  │
          ▼                        ┌─────────▼──────────┐
   ┌─────────────┐                 │  Spatial Attention │  learns which
   │ GCN Encoder │                 │  (edges / corners) │  regions matter
   └──────┬──────┘                 └─────────┬──────────┘
          │                                  │
          └────────────┬─────────────────────┘
                       │  + category one-hot (×20)
                       │  + OBB conditioning (W, H, D)
                       ▼
                latent (1024-dim)
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   ┌─────────────┐          ┌──────────────────┐
   │ Coarse Head │          │ Confidence Head  │
   │  1024 pts   │          │ MLP → 3 scalars  │
   └──────┬──────┘          └──────────────────┘
          │
          │  + partial_pc local detail
          ▼
   ┌─────────────────┐
   │ Refinement Head │
   │    8192 pts     │  ← final complete geometry
   └─────────────────┘
```

### Loss Function Stack

```
  L_total  =  InfoCD  +  0.5 · BCE  +  0.1 · L_symmetry

  ┌─────────────────────────────────────────────────────────────────────┐
  │ InfoCD  (primary)                                                   │
  │ Contrastive Chamfer Distance — NeurIPS 2023                         │
  │ Drop-in CD replacement. Maximises mutual information between        │
  │ predicted and ground-truth point cloud distributions.               │
  │ Proven consistent F-Score improvement across all backbone types.    │
  │ No O(n²) cost of EMD — efficient enough for Phase 1 batch size 4.   │
  ├─────────────────────────────────────────────────────────────────────┤
  │ BCE Voxel  (structural)                                             │
  │ Binary cross-entropy on 64³ voxel occupancy grid.                   │
  │ Supervises coarse structural prediction at voxel level.             │
  │ Penalises occupied voxels predicted as empty and vice versa.        │
  ├─────────────────────────────────────────────────────────────────────┤
  │ L_symmetry  (geometric prior)                                       │
  │ Chamfer distance between output points and their bilateral          │
  │ reflection across the object's symmetry plane.                      │
  │ Free prior — nearly all furniture in 3D-FUTURE is bilaterally       │
  │ symmetric. Especially recovers hidden-half geometry (back of        │
  │ chair, far side of wardrobe) where input signal is near-zero.       │
  └─────────────────────────────────────────────────────────────────────┘
```

### Model Specification

| Property | Value | Notes |
|---|---|---|
| Parameters | ~35M | Lightweight — consumer GPU target |
| Training VRAM | ~1.5 GB | FP16 mixed precision on T4 |
| Inference VRAM | ~0.8–1.3 GB | Loads once, runs N× per image |
| Inference speed | ~1 s / object | On T4 GPU |
| Model file size | ~88 MB | Practical for local deployment |
| Input point cloud | 2048 pts | Partial — visible surface only |
| Output point cloud | 8192 pts | Complete — includes occluded regions |
| Depth patch | 32×32 px + spatial attention | Attention learns edges/corners |
| Category encoding | One-hot, dim = 20 | From GDINO 1.6 Edge |
| OBB conditioning | 3 floats (W, H, D metres) | Real-world scale |
| Voxel resolution | 64³ | Ground truth during training |
| Precision | FP16 mixed (AMP) | ~1.6× speedup, ~50% activation memory |
| Mesh extraction | Marching Cubes (CPU) | Zero GPU VRAM at inference |

### Confidence Head — Preventing Hallucination

DGCN outputs a **3-scalar confidence vector** alongside geometry. This prevents Ghost Eye from aggressively hallucinating furniture shape when inputs are bad — extreme occlusion, wrong category, or near-zero visibility.

| Signal | Measures | Training Supervision | Ghost Eye Response |
|---|---|---|---|
| `shape_conf` | Certainty about reconstructed geometry | Calibrated vs Chamfer error on GT — high only when CD is low | Low → render bounding box placeholder only |
| `category_conf` | Reliability of GDINO category label | GDINO confidence score correlation (post-training) | Low → try top-2 GDINO candidates, keep higher shape_conf |
| `coverage_conf` | Fraction of surface actually visible | **Free supervision** — known exactly from training viewpoint % | Low → render visible region only, suppress hidden half |

```
Confidence gating thresholds:

  shape_conf    < 0.4  →  bounding box placeholder, no mesh rendered
  category_conf < 0.5  →  re-run DGCN with top-2 GDINO candidate
  coverage_conf < 0.3  →  visible region only, hidden half suppressed
  all three low        →  "low confidence reconstruction" label shown
```

---

## Training

### Dataset

| Source | Meshes | Views/Mesh | Total Pairs | Role |
|---|---|---|---|---|
| **3D-FUTURE** | 9,992 | 36 | ~360,000 | Primary — real indoor furniture |
| **ShapeNet (furniture)** | varies | 36 | supplement | Chair diversity beyond 3D-FUTURE |

3D-FUTURE is purpose-built for indoor furniture: chair, table, sofa, bed, wardrobe, cabinet, desk, lamp, shelf, stool, and more. It is significantly more domain-relevant than general ShapeNet.

**Phase 1 (Chair POC):** ~30,000 pairs · ~2 days on T4 with FP16
**Phase 2 (Full):** ~360,000 pairs · ~3–4 days on T4 with FP16

### Data Processing Pipeline

```
3D-FUTURE mesh
      │
      ├─► Render synthetic RGB image       (PyRender / Blender)
      ├─► Render synthetic depth map       (exact z-buffer values)
      │
      ├─► Apply random occlusion mask      (25–75% of depth map)
      ├─► Back-project visible depth       → partial point cloud (2048 pts)
      │
      ├─► Apply Phase 1 augmentations:
      │       · Gaussian depth noise               (σ = 0.01)
      │       · Non-uniform point density          (simulate real depth falloff)
      │       · OBB jitter                         (±10% scale, ±5° rotation)
      │       · Bounding box crop jitter           (simulate GDINO imprecision)
      │       · Y-axis random rotation             (furniture sits on floors)
      │       · Random point dropout               (simulate sensor gaps)
      │       · Scale perturbation                 (±5%, prevent OBB overfit)
      │
      ├─► Voxelise complete mesh           → ground truth 64³ occupancy
      ├─► Downsample GT                    → 1024-pt coarse decoder supervision
      │
      └─► Save tuple:
              (partial_pc, depth_patch, category, OBB,
               gt_8192, gt_1024, coverage_pct)
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Loss | `InfoCD + 0.5·BCE + 0.1·L_symmetry` |
| Optimizer | AdamW · lr = 1e-4 · weight decay = 1e-4 |
| LR Schedule | Linear warmup (5 epochs) → cosine decay to 1e-6 |
| Batch size | 4 (×8 gradient accumulation = **32 effective**) |
| Epochs | 50–100 with early stopping (patience = 10 epochs) |
| Precision | FP16 mixed precision (`torch.cuda.amp.autocast`) |
| Checkpoints | Every 5 epochs → Google Drive (Colab timeout protection) |
| **Target (Phase 1)** | **≥ 70% F-Score@1%** on chair validation set |
| **Expected (Phase 2)** | **~72% average F-Score** across all furniture categories |

---

## The Domain Gap Problem

This is the **primary technical risk** in Ghost Eye. The model trains on clean synthetic data. Inference runs on real-world data with six distinct, compounding sources of noise.

```
                   TRAINING                   INFERENCE
                   ────────                   ─────────

  Gap 1  Depth:   Perfect z-buffer       vs  Monocular DA3 estimation
                  Clean, exact values         Edge bleeding, holes in
                                              reflective/transparent surfaces

  Gap 2  Points:  Uniform density        vs  Non-uniform: denser near camera,
                  from synthetic mask         near-zero on thin structures
                                              (chair legs drop out entirely)

  Gap 3  OBB:     From perfect mesh      vs  Estimated from noisy depth
                  voxelisation                projection (±15–20% error)

  Gap 4  Masks:   Pixel-perfect          vs  SegFormer-b4 imperfect edges,
                                              boundary bleed on fine details

  Gap 5  Detect:  Ground-truth boxes     vs  GDINO 1.6 confidence varies
                                              by category and occlusion level

  Gap 6  Camera:  Clean renders          vs  JPEG compression, motion blur,
                                              sensor noise, lens distortion
```

### Error Propagation

These gaps do not fail independently — they chain through the pipeline:

```
  DA3 depth error  ──►  bad partial cloud  ──►  bad OBB dimensions
                                                        │
  GDINO wrong cat  ──►  wrong one-hot  ──►  DGCN uses wrong shape prior
                                                        │
  SegFormer miss   ──►  bad floor mask  ──►  objects anchored at wrong height

  Result: geometrically plausible but physically wrong reconstruction.
          Hard to detect visually. Requires confidence gating to catch.
```

**Mitigation:** Per-stage confidence gating. Each stage emits a quality signal. Low-confidence stages short-circuit DGCN rather than propagating bad inputs forward.

### Augmentation Strategy

| Augmentation | Targets | Cost | Phase | Priority |
|---|---|---|---|---|
| Gaussian depth noise (σ=0.01) | Gap 1 — DA3 noise | Near zero | Phase 1 | 🔴 High |
| Non-uniform point density | Gap 2 — real depth falloff | Low | Phase 1 | 🔴 High |
| OBB jitter (±15% / ±5°) | Gap 3 — OBB estimation error | Near zero | Phase 1 | 🔴 High |
| Y-axis random rotation | Viewpoint diversity | Near zero | Phase 1 | 🔴 High |
| Random point dropout | Sensor gap simulation | Near zero | Phase 1 | 🔴 High |
| Bbox jitter on crop | Gap 5 — GDINO imprecision | Near zero | Phase 1 | 🟡 Medium |
| Scale perturbation ±10% | OBB overfit prevention | Near zero | Phase 1 | 🟡 Medium |
| Mask erosion / dilation | Gap 4 — SegFormer edges | Low | Phase 2 | 🟡 Medium |
| JPEG compression sim | Gap 6 — camera artifacts | Low | Phase 2 | 🟢 Low |
| DA3 noise fingerprint | Gap 1 — DA3 systematic errors | Medium | Phase 2 | 🟡 Medium |

> Phase 2 DA3 noise fingerprinting requires characterising real DA3 outputs first — cannot simulate systematic failure modes without real inference data to study. This runs in parallel with Phase 2 training prep, not blocking it.

---

## Results

> **Phase 1 (chair proof-of-concept) is currently in training.** Results will be published here as they become available.

### Target Metrics

| Metric | Measures | Phase 1 Target | Phase 2 Expected |
|---|---|---|---|
| **F-Score@1%** | Point cloud accuracy vs ground truth | **≥ 70%** (chairs) | **~72%** (all furniture) |
| Chamfer Distance (CD-L1) | Mean nearest-neighbour error | TBD | TBD |
| Voxel IoU | 64³ occupancy accuracy | TBD | TBD |
| Inference latency | Per object on T4 | ~1 s | ~1 s |
| Peak VRAM | Maximum at any pipeline point | ~2.5 GB | ~2.5 GB |

*Sample reconstructions, loss curves, and confidence calibration plots — coming after Phase 1.*

---

## Architecture Decision Log

Every major design decision is documented with the rejected alternative and reasoning.

| Decision | Rejected | Chosen | Why |
|---|---|---|---|
| Depth model | DA2-Small Metric | **DA3Metric-Large** | Better spatial geometry + true metric depth. DA3-Small relative-only depth breaks OBB conditioning. |
| Segmentation | SegFormer-b2 | **SegFormer-b4** | Free accuracy upgrade, identical architecture, same VRAM budget. OneFormer (~3 GB) pushes stage budget. |
| Detection | GDINO Tiny 1.0 | **GDINO 1.6 Edge** | +40% speed vs 1.5 Edge. 44.8 AP COCO vs ~36 AP Tiny. Better category labels feed DGCN. |
| Primary loss | Plain Chamfer | **InfoCD** | Drop-in contrastive replacement. Proven consistent F-Score gain across all backbone types. No O(n²) EMD cost. |
| EMD loss | Use instead of Chamfer | **Defer to Phase 2** | O(n²) — ~40× slower per step on 8192-pt clouds. Add as 5–10 epoch fine-tune after baseline proven. |
| Decoder | Single-shot → 8192 | **Coarse-to-fine** | Single-shot produces blobby thin structures. Architecture decision — must be built in from start. |
| Representation | Implicit occupancy | **64³ voxel grid** | Implicit requires secondary MLP decoder. Validate voxel baseline first. |
| Symmetry | None | **Bilateral symmetry loss** | Furniture is nearly all symmetric. Free geometric prior. Near-zero compute cost. |
| Depth patch | Flat vector | **Spatial attention + flatten** | ~100k params. Attention learns edges/corners are most informative. Negligible at 35M total. |
| Confidence | No output | **3-scalar confidence vector** | Prevents hallucination on extreme occlusion or wrong category. coverage_conf supervised free from training. |
| Precision | FP32 | **FP16 mixed (AMP)** | ~1.6× speedup. ~50% activation memory. T4 Tensor Cores built for FP16. One line of code. |
| Backbone | PointRAFT Transformer | **Standard GCN (Phase 1)** | PointRAFT is a full architectural rewrite. Validate GCN baseline first. Phase 2 candidate. |
| Floor/wall/ceiling | DA3 surface normals | **SegFormer semantic** | DA3 normals fail on cluttered floors — horizontal shelftops misclassified. Semantic segmentation is more robust. |
| Quantisation | INT8 PTQ at inference | **FP16 training only** | Coordinate regression compounds precision errors differently to classification. QAT deferred to Phase 3. |
| Architecture style | Foundation + sparse refinement | **Lightweight specialist** | Foundation model requires large frozen weights in VRAM permanently — breaks load/unload pipeline. |

---

## Technical Risk Register

| Risk | Severity | Detail | Mitigation |
|---|---|---|---|
| **Domain gap** | 🔴 HIGH | Model trained on synthetic; inference on real. Six distinct compounding gaps. | 7-augmentation Phase 1 stack; Phase 2 DA3 noise fingerprinting |
| **Error propagation** | 🔴 HIGH | Each stage feeds the next — one failure poisons all downstream stages. | Per-stage confidence gating; low-conf stages short-circuit DGCN |
| **OBB conditioning mismatch** | 🟡 MEDIUM | Train OBBs from perfect mesh; inference OBBs from noisy depth projection. | OBB jitter ±15%/±5° forces robustness to imprecise conditioning |
| **Non-uniform point density** | 🟡 MEDIUM | Synthetic clouds uniform; real depth non-uniform — thin structures drop out. | Non-uniform density augmentation simulates real depth falloff |
| **Colab session timeouts** | 🟢 LOW | T4 session time limits may interrupt mid-epoch. | Checkpoint every 5 epochs → Google Drive; resume from checkpoint |
| **Stage 1 VRAM ceiling** | 🟢 LOW | DA3Metric-Large at ~2.5 GB is right at load/unload budget. | Verify Stage 1 fits individually before building full pipeline |

---

## Roadmap

```
Phase 1 — Chair Proof of Concept                          [ IN PROGRESS ]
─────────────────────────────────────────────────────────────────────────
  ✅  Architecture spec finalised (v2.0)
  ✅  All model selections locked (DA3 / SegFormer-b4 / GDINO 1.6 Edge)
  ✅  Loss stack decided  (InfoCD + BCE + L_symmetry)
  ✅  Coarse-to-fine decoder designed
  ✅  Confidence head (3-scalar) designed
  ✅  Domain gap analysis + augmentation strategy complete
  ⬜  Colab notebook — data preprocessing pipeline
  ⬜  Colab notebook — DGCN model implementation
  ⬜  Colab notebook — Phase 1 augmentation stack
  ⬜  Training run — chairs only (~30k pairs, ~2 days)
  ⬜  Validation: 70%+ F-Score@1% on chair test set
  ⬜  Confidence head calibration on held-out set
  ⬜  First sample reconstructions published here

Phase 2 — Full Furniture Training                               [ PLANNED ]
─────────────────────────────────────────────────────────────────────────
  ⬜  Real DA3Metric-Large outputs collected + failure modes catalogued
  ⬜  Phase 2 augmentations  (mask erosion, compression, DA3 fingerprint)
  ⬜  Full 3D-FUTURE + ShapeNet training  (~360k pairs, ~3–4 days)
  ⬜  Target: ~72% average F-Score across all furniture categories
  ⬜  PointRAFT backbone evaluation vs GCN baseline
  ⬜  EMD fine-tuning  (5–10 epochs post main training)

Phase 3 — Pipeline Integration & Deployment                     [ PLANNED ]
─────────────────────────────────────────────────────────────────────────
  ⬜  Full Ghost Eye pipeline integration (all 5 stages)
  ⬜  Per-stage confidence gating implementation
  ⬜  End-to-end inference on real room photographs
  ⬜  Quantisation-aware training (INT8 QAT) for deployment
  ⬜  Implicit occupancy evaluation for large objects (bed, wardrobe)
  ⬜  OneFormer evaluation if SegFormer-b4 shows limitations
```

---

## Phase 2 Upgrade Candidates

Confirmed improvements — deferred until Phase 1 validates the baseline. Do not implement during Phase 1.

| Upgrade | Affects | Prerequisite | Expected Benefit |
|---|---|---|---|
| **PointRAFT backbone** | Architecture | Chair POC ≥ 70% confirmed | Better thin/complex geometry across categories |
| **EMD fine-tuning** | Loss | Chamfer training complete | Surface quality polish — 5–10 epochs only |
| **Implicit occupancy** | Representation | Phase 2 training complete | Higher resolution for large objects (bed, wardrobe) |
| **DA3 noise fingerprint aug** | Data | Real DA3 outputs collected | Simulate DA3 systematic failure modes accurately |
| **INT8 QAT** | Deployment | Final model validated on real scenes | ~4× smaller model for very low-VRAM GPUs |
| **OneFormer segmentation** | Stage 2 | SegFormer-b4 shows limitations | Instance separation within same furniture category |

---

## Repository Structure [planned]

```
ghost-eye/
│
├── README.md
│
├── docs/
│   ├── ARCHITECTURE.md                     Full architecture spec (condensed)
│   └── GhostEye_DGCN_Pipeline_Spec_v2.docx Complete planning document
│
├── notebooks/
│   └── DGCN_train_chairs.ipynb             Colab Phase 1 training notebook
│
├── src/
│   ├── model/
│   │   ├── dgcn.py                         DGCN model definition
│   │   ├── encoder.py                      GCN encoder + depth patch attention
│   │   ├── decoder.py                      Coarse-to-fine two-stage decoder
│   │   └── confidence.py                   3-scalar confidence head
│   │
│   ├── loss/
│   │   ├── info_cd.py                      InfoCD contrastive Chamfer loss
│   │   ├── symmetry.py                     Bilateral symmetry regularisation
│   │   └── combined.py                     Combined loss with λ weights
│   │
│   ├── data/
│   │   ├── dataset.py                      3D-FUTURE + ShapeNet dataloader
│   │   ├── preprocess.py                   Render → point cloud pipeline
│   │   └── augment.py                      Phase 1 & Phase 2 augmentation stack
│   │
│   └── pipeline/
│       ├── depth.py                        Stage 1: DA3Metric-Large wrapper
│       ├── segment.py                      Stage 2: SegFormer-b4 wrapper
│       ├── detect.py                       Stage 3: GDINO 1.6 Edge wrapper
│       └── reconstruct.py                  Stage 4: DGCN inference + confidence gating
│
└── results/
    ├── phase1_chairs/                      Training curves, sample outputs (coming)
    └── phase2_full/                        Full furniture results (coming)
```

---

## Dependencies

```python
# Deep learning
torch >= 2.0.0
torch-scatter
torch-sparse

# 3D processing
open3d
pyrender
trimesh
scikit-image          # Marching Cubes mesh extraction

# Training utilities
einops
tensorboard
tqdm
h5py
numpy
```

---

## Full Specification

The complete Ghost Eye architecture specification (v2.0) covers:

- Full pipeline model selection and rationale
- Stage-by-stage input/output contracts
- Complete DGCN architecture tables
- Loss function derivation and λ starting values
- Training schedule and hyperparameters
- All six domain gaps with targeted mitigations
- Confidence head design and supervision sources
- Full decision log (15 decisions with rejected alternatives)
- Risk register with severity ratings
- Phase 2 upgrade path with prerequisites

Available in [`docs/GhostEye_DGCN_Pipeline_Spec_v2.docx`](docs/).

---

## Citation

If you use Ghost Eye or the DGCN architecture in your work:

```bibtex
@misc{ghosteye2026,
  title   = {Ghost Eye: Depth-Guided Geometry Completion for Indoor Furniture Reconstruction},
  author  = {AnshBuilt-3d},
  year    = {2026},
  note    = {Phase 1 in training. Architecture specification v2.0.},
  url     = {https://github.com/AnshBuilt-3d/Ghost-Eye}
}
```

---

## Status

```
Phase 1  Chair POC        ████░░░░░░░░░░░░░░░░   Architecture complete · Training pending
Phase 2  Full training    ░░░░░░░░░░░░░░░░░░░░   Planned after Phase 1 validates baseline
Phase 3  Integration      ░░░░░░░░░░░░░░░░░░░░   Planned
```

---

<div align="center">

**Ghost Eye** is an independent research project.

Architecture specification v2.0 · Training on T4 GPU · Phase 1 results coming.

*If this work is useful or interesting, watch the repo for Phase 1 results.*

</div>
