# AdaptFit: Implementation Plan and Master Roadmap Toward the Congressional App Challenge

> **Documentation metadata**
> - **Status:** canonical-active
> - **Authority:** exhaustive dependency-ordered engineering master plan and delivery roadmap
> - **Last verified:** 2026-09-07
> - **Source commit:** `ba8bf1a`
> - **Owner:** AdaptFit engineering & machine learning team
> - **Supersedes or supports:** canonical forward plan and engineering roadmap; supersedes earlier high-level implementation outlines
> - **Review trigger:** milestone completion, dependency change, gate result, or empirical evidence

Planning review: September 7, 2026. Audited model checkout: `ba8bf1a`.

---

## Executive Summary & Product Vision

AdaptFit is an anatomy-informed, on-device movement adaptation and repetition tracking system designed specifically for individuals with bodily limitations:
1. **Upper-limb differences / single-arm users** (transradial/transhumeral amputations, hemiparesis, brachial plexus injuries).
2. **Lower-limb differences / single-leg users** (transtibial/transfemoral amputations, single-leg functional mobility).
3. **Wheelchair and seated users** (manual wheelchair users, spinal cord injuries, seated fitness).

The core product delivers an autonomous, offline workout loop:
$$\text{Self-Reported Capabilities \& Equipment} \longrightarrow \text{Compatible Routine Assembly} \longrightarrow \text{Camera-Assisted Exercise} \longrightarrow \text{Causal Repetition Tracking \& Form Feedback} \longrightarrow \text{Equipment Substitution} \longrightarrow \text{Verified Progress Logging}$$

The first-release scope covers **five seated, unilateral-friendly rehabilitation exercises**:
- One-Arm Bicep Curl (unilateral upper-arm flexion)
- One-Arm Band Row (unilateral scapular retraction & pull)
- Seated Single-Leg Knee Extension (unilateral lower-leg extension)
- Seated Single-Leg March (unilateral hip flexion)
- Seated Forward Reach (unilateral/bilateral functional reaching & trunk stabilization)

---

## What Exists vs. What Remains Unproven

| Subsystem | Verified Current State | Identified Empirical Bottleneck | Master Plan Solution & Next Action |
|---|---|---|---|
| **Pose Landmarks** | Canonical 33-joint 2D layout normalized around hip midpoint and torso scale | Optical mocap datasets (Vicon/Qualisys) are in 3D millimeters | Standardized 3D-to-2D virtual camera projection pipeline |
| **Feature Pipeline** | 283 ordered features (`training/src/features/anatomy.py`): angles, velocities, masks, capability context | Normalization assumes intact bilateral body by default | Explicit capability weights ($w_c \in \{1.0, 0.8, 0.65, 0.5, 0.0\}$) modulate confidence and zero out absent anatomy |
| **Model Architectures** | Causal Dilated TCN (307,410 params; 5 blocks); Causal GRU (68,178 params; 1 layer) | TCN receptive field is strictly 125 frames (~4.16s at 30 FPS) | Separate long-horizon count state machine from fixed-context backbone |
| **Repetition Counting** | Repetition Start F1: **66.96%–99.54%**; Repetition End F1: **12.26%–18.47%** | Naive per-frame accumulation $\sum p_{end}(t)$ causes severe overcounting | **Phase 1**: Phase-Coupled Finite State Machine with dual hysteresis |
| **Training Pipeline** | Overnight 72-epoch TCN trained; GRU interrupted at epoch 52 | Full retraining requires 18 hrs across 267k windows; no warm-start | **Phase 2**: Event-weighted sampler ($4.4\times$ speedup) + warm-start runner |
| **Quality Supervision** | 4-dimensional quality heads (`quality_logits`) have **0% labeled coverage** in v1/v2 | UCO composite score is 1-5 scalar, not dimension-specific | **Phase 3**: Ingest SERE, TRSPD, and KERAAL clinical compensation labels (UI-PRMD correctness remains masked from dimension-specific quality heads) |
| **Teacher Distillation** | Supervised baseline only; no teacher models integrated | Boundary jitter on variable-cadence repetitions | **Phase 4**: SSTRAC repetition density + PoseRAC salient-state distillation |
| **Mobile Deployment** | Python causal streaming runtime (`CausalStreamingRuntime`) passing unit tests | Native Android/iOS MediaPipe feature bridge and on-device runtime unbuilt | **Phase 5**: ONNX / TFLite INT8 export + 6 golden fixture parity tests |
| **Target Validation** | Tested on public datasets (REHAB24-6, IntelliRehabDS, MM-Fit, UL-RED, UCO) | Zero validation on real amputee, wheelchair, or stroke participants | **Phase 6**: Consented $N=10\text{–}15$ pilot study + Congressional App Challenge delivery |

---

## The Exhaustive 6-Phase Engineering Master Roadmap

```mermaid
flowchart TD
    P1[Phase 1: Decoder Calibration & Event Debouncing] --> P2[Phase 2: Warm-Start & Fast Training Runner]
    P2 --> P3[Phase 3: Clinical & Target-Population Dataset Ingestion]
    P3 --> P4[Phase 4: Repetition Density & Teacher Distillation]
    P4 --> P5[Phase 5: Mobile Runtime Export & Golden Parity]
    P5 --> P6[Phase 6: Consented Pilot Validation & CAC Submission]

    style P1 fill:#1e3a8a,stroke:#3b82f6,color:#ffffff
    style P2 fill:#1e3a8a,stroke:#3b82f6,color:#ffffff
    style P3 fill:#1e3a8a,stroke:#3b82f6,color:#ffffff
    style P4 fill:#1e3a8a,stroke:#3b82f6,color:#ffffff
    style P5 fill:#1e3a8a,stroke:#3b82f6,color:#ffffff
    style P6 fill:#065f46,stroke:#10b981,color:#ffffff
```

---

### Phase 1: Decoder Calibration & Event Debouncing (Task C1)

**Owner: Machine Learning Team | Priority: Immediate (Sprint 1)**  
**Status: Ready for Implementation | Prerequisites: Evaluated `v2-quality-fixed` checkpoint**

#### 1. Root Cause Diagnosis
In `training/src/metrics.py#L285-L287`, the current repetition counter computes counts by summing per-frame boundary logits:
```python
# Flawed accumulator in metrics.py
pred_counts = (seq_pred[:, 1] > threshold).sum().item()
```
Because the causal TCN emits positive end logits across consecutive frames (e.g. 5–15 frames during the eccentric phase inflection), this raw accumulation causes catastrophic overcounting: a single 10-repetition workout set can register 80–150 false repetitions.

#### 2. Phase-Coupled Finite State Machine (FSM) Specification
To enforce biological periodicity, the counting decoder is formulated as an asynchronous finite state machine coupled with the model's per-frame phase probabilities:

```mermaid
stateDiagram-v2
    [*] --> IDLE_REST
    IDLE_REST --> CONCENTRIC_DRIVE: p_start >= tau_start_high AND phase == CONCENTRIC
    CONCENTRIC_DRIVE --> APEX_HOLD: phase == HOLD OR delta_angle == 0
    APEX_HOLD --> ECCENTRIC_RETURN: phase == ECCENTRIC
    ECCENTRIC_RETURN --> CYCLE_COMPLETE: p_end >= tau_end_high AND delta_t >= T_ref
    CYCLE_COMPLETE --> IDLE_REST: Emit WorkoutEventV1(RepComplete)
    
    CONCENTRIC_DRIVE --> IDLE_REST: Aborted Repetition (Timeout / Reversal)
    IDLE_REST --> PAUSED: p_track < tau_track (Occlusion / Tracking Loss)
    PAUSED --> IDLE_REST: p_track >= tau_track (Tracking Restored)
```

#### 3. Mathematical State Transition Rules
Let $p_{start}(t)$, $p_{end}(t)$, $p_{phase}(t) \in [0, 1]^5$, and $p_{track}(t) \in [0, 1]$ denote the model outputs at frame $t$:

1. **Dual-Threshold Hysteresis for Start Trigger**:
   - Transition from `IDLE_REST` to `CONCENTRIC_DRIVE` requires:
     $$p_{start}(t) \ge \tau_{start\_high} \quad (0.65) \quad \text{AND} \quad \arg\max p_{phase}(t) \in \{\text{concentric}, \text{hold}\}$$
   - Hysteresis releases only when $p_{start}(t) < \tau_{start\_low} \quad (0.35)$.

2. **Dual-Threshold Hysteresis for End Trigger**:
   - Transition from `ECCENTRIC_RETURN` to `CYCLE_COMPLETE` requires:
     $$p_{end}(t) \ge \tau_{end\_high} \quad (0.45) \quad \text{AND} \quad \arg\max p_{phase}(t) \in \{\text{eccentric}, \text{rest}\}$$
   - Hysteresis releases only when $p_{end}(t) < \tau_{end\_low} \quad (0.25)$.

3. **Temporal Refractory Period ($T_{ref}$)**:
   - A minimum repetition cycle time $T_{ref} = 24\text{ frames}$ ($0.80\text{ seconds}$ at 30 FPS) is strictly enforced. If $t_{end} - t_{start} < T_{ref}$, the event is classified as motion jitter and discarded.

4. **Local Peak Filter**:
   - The end event is emitted only at the local temporal maximum within a sliding window of radius $W=2$ frames:
     $$p_{end}(t) = \max_{k \in [-2, +2]} p_{end}(t+k)$$

5. **Tracking Observability Gating**:
   - If tracking confidence $p_{track}(t) < \tau_{track} = 0.60$, the state machine enters `PAUSED`. State transitions are frozen, preventing spurious counts during camera occlusion or framing loss. When tracking recovers, a 5-frame stabilization warmup is enforced before resuming.

#### 4. Grid-Search Optimization Protocol
- **Search Space**:
  - $\tau_{start\_high} \in [0.50, 0.85]$ (step 0.05)
  - $\tau_{end\_high} \in [0.30, 0.65]$ (step 0.05)
  - $T_{ref} \in [15, 36]$ frames (step 3 frames / 0.1s)
  - $\tau_{track} \in [0.40, 0.75]$ (step 0.05)
- **Validation Target**: Replay inference over all 16,483 validation windows prepared in `data/processed-v2-quality/` (`validation/*.npy` memmaps and `validation.jsonl`) using `artifacts/v2-quality-fixed/checkpoints/tcn_best.pt`. (Note: `artifacts/v2-quality-fixed/evaluations/tcn_validation.json` does not exist; existing evaluation artifacts in `artifacts/v2-quality-fixed/metrics/tcn_evaluation.json` represent test split results only). The 7,536-window (1,007-sequence) test split must remain strictly locked and unpeeked during decoder calibration to prevent evaluation leakage and protect test set integrity.
- **Exit Gate**: Repetition Count MAE $\le 0.40$ (reducing validation count error by $>80\%$) and Sequence-Level Repetition End F1 $\ge 60.0\%$ evaluated on the validation split.

---

### Phase 2: Warm-Start & Fine-Tuning Runner Implementation (Tasks B1–B4)

**Owner: Machine Learning Team | Priority: High (Sprint 1)**  
**Status: Pending (blocked on Phase 1 calibration) | Prerequisites: Phase 1 calibration**

#### Task B1: Safe Warm-Start Weight Loading
Implement `load_pretrained_backbone` in `training/src/runner.py`:
```python
def load_pretrained_backbone(model: nn.Module, checkpoint_path: str | Path, freeze_backbone: bool = False) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_dict = model.state_dict()
    
    loaded_keys, skipped_keys = [], []
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)
            
    model.load_state_dict(model_dict, strict=False)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not any(head_name in name for head_name in ["heads", "density"]):
                param.requires_grad = False
                
    return {"loaded": len(loaded_keys), "skipped": len(skipped_keys)}
```

#### Task B2: Trainable-Layer Selection & Exact Resumption Checkpoints
- Support three fine-tuning regimes via `--fine-tune-mode`:
  1. `heads_only`: Freezes all 5 TCN dilated residual blocks; updates only output heads (fast adaptation in <30 min).
  2. `last_block`: Freezes TCN blocks 0–3; updates residual block 4 and all output heads (preserves low-level kinematic primitives while adapting high-level temporal dynamics).
  3. `full_backbone`: Discriminative learning rates ($2 \times 10^{-5}$ for backbone, $1 \times 10^{-4}$ for heads).
- Checkpoint serialization stores: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `best_metric_score`, `epoch`, and `label_policy_version`.

#### Task B3: Redundant Work Reduction via Event-Weighted Sampler
- **Bottleneck**: `data/processed-v2-quality/` contains 267,440 training windows. An audit reveals that $>65\%$ of these windows capture steady-state rest periods between exercise sets.
- **Solution**: Implement `EventWeightedWindowSampler`.
  - Windows within $t \in [t_{start} - 8, t_{end} + 8]$ (active repetitions) sampled at stride 4.
  - Windows during sustained rest sampled at stride 32.
  - Reduces active training volume from 267k windows to ~60k windows.
  - **Yields a $4.4\times$ reduction in per-epoch training time** (from 15 minutes/epoch down to 3.4 minutes/epoch on Apple Silicon MPS, completing a 100-epoch run in ~5.5 hours instead of 25 hours).

#### Task B4: Bounded Experiment Configuration
Create `training/configs/experiments/warmstart_fine_tune.yaml` with explicit learning rate schedules, gradient clipping ($1.0$), and early stopping patience (15 epochs on validation count MAE).

---

### Phase 3: Clinical & Target-Population Dataset Ingestion

**Owner: Data Engineering Team | Priority: High (Sprint 2)**  
**Status: Blueprint Specified | Authority: `docs/dataset-catalog.md` & `docs/dataset-expansion-plan.md`**

```mermaid
graph LR
    subgraph Tier1[Tier 1: Immediate Ingestion]
        D1[DynTherapy: 33-pt 1:1] --> AD1[training/src/data/adapters.py]
        D2[UI-PRMD: 3D to 2D Mocap] --> AD1
        D3[Pipelines: Wheelchair 2D/3D] --> AD1
        D4[ROAG: Transradial Mocap] --> AD1
        D5[Ottobock: Amputee Visuals] --> AD2[Pose Front-End Shards]
    end
    subgraph Tier2[Tier 2: Clinical DUAs]
        D6[SERE: Stroke Compensations] --> AD1
        D7[TRSPD / TULE: Stroke Pose] --> AD1
        D8[MobiPhysio: Mobile Smartphone] --> AD1
        D9[InclusiveVidPose: Limb Difference] --> AD2
    end
```

#### Step-by-Step Acquisition Sequence

1. **DynTherapy (Priority 1, Sprint 1)**:
   - Ingest 33 MediaPipe keypoints from Mendeley Data (CC BY 4.0).
   - Direct 1:1 drop-in adapter in `training/src/data/adapters.py#load_dyntherapy`.
   - Supplies verified repetition start/end boundaries and phase labels across 7 PT exercises.

2. **UI-PRMD (Priority 1, Sprint 1)**:
   - Ingest 10 PT exercises with dual Vicon/Kinect data.
   - Project 3D markers to canonical 2D via virtual camera projection ($d=2.0\text{m}, h=1.0\text{m}$).
   - Maps repetition cycle boundaries and exercise family labels; dimension-specific quality heads remain strictly masked (`quality_mask = 0.0`, target `-1`). UI-PRMD provides overall movement correctness ("optimal" vs. "non-optimal"), which must NOT be mapped into dimension-specific ROM (`quality[:, 0]`) or trunk compensation (`quality[:, 3]`) heads per the feature contract and `dataset-expansion-plan.md`.

3. **Pipelines Open Dataset (Priority 1, Sprint 1)**:
   - Ingest synchronized wheelchair propulsion cycles.
   - Maps push/recovery timestamps to concentric/eccentric phases.
   - Establishes wheelchair camera-to-mocap benchmark.

4. **ROAG (Priority 1, Sprint 1)**:
   - Ingest 2 transradial amputee reaching trajectories (2,450 trials).
   - Maps reach geometry and torso lean to single-arm capability profiles and thresholded binary trunk compensation quality (applying an explicit biomechanical threshold $\theta_{trunk} \ge \tau_{trunk} = 15^\circ$ to prevent a schema mismatch with the binary classification head `quality_logits[:, 3]`).

5. **Ottobock #DearAI Community Library (Priority 1, Sprint 1)**:
   - Ingest community imagery of upper-limb and lower-limb amputees.
   - Evaluates MediaPipe landmark confidence and capability weights ($w_c$) to prevent phantom limb hallucination.

6. **SERE & TRSPD / TULE (Priority 2, Sprint 2)**:
   - Submit research DUA to VisLab Lisbon (`ana.coias@tecnico.ulisboa.pt`).
   - Ingest 18–20 post-stroke 3D skeletons with therapist-graded trunk lean and shoulder hiking annotations, populating the trunk compensation head.

---

### Phase 4: Repetition Density & Teacher Distillation

**Owner: Machine Learning Research | Priority: Medium (Sprint 3)**  
**Status: Architecture Designed | Candidates: SSTRAC, PoseRAC, RACNet, MotionBERT**

#### 1. Student Repetition Density Head Architecture
Extend `MultiTaskHeads` in `training/src/models/heads.py`:
```python
# In MultiTaskHeads.__init__:
self.density = nn.Linear(channels, 1)

# In MultiTaskHeads.forward:
density_raw = self.density(sequence).squeeze(-1)  # Shape: [batch, time]
outputs["density"] = F.softplus(density_raw)      # Guarantees non-negative density mass
```

#### 2. Continuous Density Loss Formulation
For a sequence window $[0, T]$ with predicted density $\hat{d}(t)$ and ground-truth density $d^*(t)$:
$$\mathcal{L}_{density} = \frac{1}{T} \sum_{t=1}^T |\hat{d}(t) - d^*(t)| + \lambda_{count} \left| \sum_{t=1}^T \hat{d}(t) - C^* \right|$$
where $C^*$ is the exact integer repetition count in the window, and $\lambda_{count} = 0.5$. Continuous repetition counting on live streams is obtained simply by integrating density over time:
$$\text{Count}(T) = \left\lfloor \int_0^T \hat{d}(t) \, dt + 0.5 \right\rfloor$$

#### 3. Teacher Distillation Roles & Interfaces

| Teacher Candidate | Reference & Repository | Input Modality & Joints | Distilled Target & Loss Function | Stage & Selection Gate |
|---|---|---|---|---|
| **SSTRAC** | Lim et al., *IEEE Access* (2025)<br>`imjjun/SSTRAC_public` | 2D/3D skeleton sequences (17 joints) | **Continuous Repetition Density Map**: Distills frame-level density $\hat{d}_{teacher}(t)$ via L1 loss | **First Candidate**: Gate: Count MAE improves over supervised baseline on RepCount-pose |
| **PoseRAC** | Yao et al., *CVPR* (2023)<br>`MiracleDance/PoseRAC` | BlazePose 33-joint sequences | **Salient-State Apex Logits**: Distills inflection point probabilities via KL-divergence | **Second Candidate**: Gate: End F1 improves without increasing count MAE |
| **RACNet** | Luo et al., *ECCV* (2024)<br>`Luoadore/RACnet` | Video feature sequences | **Cycle-Start Probabilities**: Distills full-resolution start-logit probabilities | **Third Candidate**: Optional ablation for cycle initialization |
| **MotionBERT** | Zhu et al., *ICCV* (2023)<br>`Walter0807/MotionBERT` | 17-joint 2D pose + confidence | **Latent Biomechanical Embeddings**: MSE loss between projected student features and frozen MotionBERT latents | **Auxiliary Only**: General motion prior; does not run on mobile |

#### 4. Offline Teacher Caching Pipeline
- Teachers are evaluated offline over training sequences. Outputs are stored in `.npz` files indexed by `sequence_id`:
  ```text
  data/teacher_cache/
  ├── sstrac_density/
  │   ├── manifest.json (model hash, git commit, timestamp)
  │   └── seq_{sequence_id}.npz (density: float32 [T])
  └── poserac_salient/
      ├── manifest.json
      └── seq_{sequence_id}.npz (salient_logits: float32 [T, 2])
  ```
- Strict split boundaries: Teacher targets are computed **only for the training split**. Validation and test splits evaluate ground-truth labels exclusively.

---

### Phase 5: Mobile Runtime Export & Golden Parity Verification

**Owner: Mobile & Runtime Team | Priority: High (Sprint 3–4)**  
**Status: Ready for Export Pipeline | Target Devices: Android (Pixel 7 / Galaxy A54), iOS (iPhone 13+)**

#### 1. Export Pipeline & Quantization Architecture
```mermaid
flowchart LR
    PyTorch[PyTorch Checkpoint<br>tcn_best.pt] --> ONNX[ONNX Export<br>opset 17]
    ONNX --> TFLite[TFLite Model<br>Float32]
    TFLite --> INT8[TFLite INT8<br>Dynamic / Static Calibration]
    INT8 --> MobileApp[AdaptFit Android/iOS<br>Native C++ / Kotlin / Swift]
```

- **Export Script**: `training/src/export/export_mobile.py` exports the causal TCN with a fixed streaming state buffer:
  - Input: Current frame features $[1, 1, 283]$ + internal causal buffer $[1, 96, 125]$.
  - Output: Multi-task predictions + updated buffer $[1, 96, 125]$.
- **Post-Training Quantization (PTQ)**: Calibrated using 1,000 representative validation windows.

#### 2. Golden Fixture Test Matrix
Six canonical test vectors are serialized to `tests/fixtures/golden_parity/`:
1. **Vector 1: Bilateral Standing Curl** (Symmetric intact movement).
2. **Vector 2: Unilateral Seated Curl** (Left arm active, right arm absent/capability zeroed).
3. **Vector 3: Unilateral Seated Band Row** (Single-arm pull with trunk stabilization).
4. **Vector 4: Seated Wheelchair Knee Extension** (Seated posture, lower-leg extension).
5. **Vector 5: Seated Wheelchair Forward Reach** (Trunk flexion and shoulder elevation).
6. **Vector 6: Camera Occlusion / Dropped Frames** (Tracking loss and recovery).

#### 3. Parity Tolerances & Runtime Benchmarks

| Metric / Dimension | Target Tolerance Gate | Failure Condition |
|---|---|---|
| **Float32 Parity** | $\max |\hat{y}_{py} - \hat{y}_{onnx}| < 1 \times 10^{-4}$ | Any logit divergence $> 1 \times 10^{-3}$ |
| **INT8 Quantized Parity** | $\max |\hat{y}_{py} - \hat{y}_{int8}| < 0.05$ | FSM event trigger mismatch on golden fixtures |
| **Event Alignment** | 100% agreement on Repetition Start and End timestamps | $\ge 1$ skipped or extraneous repetition event |
| **Camera-to-Display Latency** | $\mathbf{p50 < 45\text{ms}}, \quad \mathbf{p95 < 95\text{ms}}$ | p95 latency $> 150\text{ms}$ at 30 FPS |
| **Memory Footprint** | Peak RAM consumption $\mathbf{< 25\text{MB}}$ | Peak RAM $> 50\text{MB}$ |
| **Thermal & Battery** | $< 15\%$ single-core CPU utilization; no thermal throttling over 30-min workout | Battery drain $> 8\%$ per 30 minutes |

---

### Phase 6: Consented Clinical / Target-User Validation & Challenge Readiness

**Owner: Product Lead & ML Reviewer | Priority: Critical (Sprint 4–5)**  
**Status: Planning & Protocol Design | Submission Target: October 23, 2026**

#### 1. Consented Pilot Study Protocol
- **Cohort Target**: $N = 10 \text{ to } 15$ diverse participants:
  - 3–5 individuals with upper-limb differences (transradial, transhumeral, brachial plexus, hemiparesis).
  - 3–5 individuals using manual wheelchairs or seated mobility devices.
  - 3–5 individuals with lower-limb differences or unilateral mobility limits.
- **Protocol**: 5 launch exercises performed in home/clinic settings under natural smartphone camera framing.
- **Ethics & Privacy**:
  - Signed informed consent covering video recording and biometric motion analysis.
  - Complete local on-device processing: Zero raw video or pose coordinates uploaded to cloud servers.
  - Option for participants to view and immediately purge motion data after session completion.

#### 2. Congressional App Challenge (CAC) Submission Roadmap

```mermaid
gantt
    title Congressional App Challenge Delivery Timeline (2026)
    dateFormat  YYYY-MM-DD
    section Phase 1-2: Core Engine
    Decoder Calibration (Task C1)     :active, 2026-09-08, 2026-09-12
    Fast Warm-Start Runner (B1-B4)    :2026-09-12, 2026-09-18
    section Phase 3-4: Data & Teachers
    Tier 1 Dataset Ingestion          :2026-09-18, 2026-09-26
    SSTRAC/PoseRAC Distillation       :2026-09-26, 2026-10-04
    section Phase 5: Mobile Runtime
    ONNX/TFLite Parity & Mobile App   :2026-09-28, 2026-10-08
    On-Device Profiling               :2026-10-08, 2026-10-14
    section Phase 6: Pilot & Submission
    Consented Pilot Validation        :2026-10-05, 2026-10-16
    Demo Video Production             :2026-10-14, 2026-10-21
    Internal Final Review             :2026-10-22, 2026-10-23
    Official CAC Deadline (12 PM EDT) :milestone, 2026-10-26, 0d
```

*Status Alignment with Execution Log*: As documented in `docs/training-execution-log.md`, Decoder Calibration (Task C1) is currently READY / active for execution. Fast Warm-Start Runner (Tasks B1–B4) is PENDING completion of calibration. Neither is marked complete, and downstream data ingestion and distillation phases are scheduled rather than active.

- **Key Dates**:
  - **Internal Submission Freeze**: **October 23, 2026** (3-day safety buffer).
  - **Official CAC Submission Deadline**: **October 26, 2026, 12:00 PM EDT**.
- **1–3 Minute Demonstration Video Storyboard**:
  - **Act 1: The Problem (0:00–0:30)**: Generic fitness apps assume two arms, two legs, and standing posture. For amputees, wheelchair users, and stroke survivors, these apps fail immediately.
  - **Act 2: The Solution & Setup (0:30–1:15)**: User profile onboarding: selecting "Single-Arm (Right)" and "Seated". AdaptFit configures a feasible 5-exercise routine and sets up smartphone camera tracking.
  - **Act 3: Live On-Device Execution (1:15–2:15)**: Real-time workout demo prototype. The user performs seated curls and band rows. Visual overlays illustrate capability-masked tracking on absent limbs, repetition counting using calibrated phase-coupled debouncing (targeting MAE $\le 0.40$), and prototype compensation monitoring (pending reviewed quality supervision).
  - **Act 4: Technical Architecture & Impact (2:15–3:00)**: Student-built causal dilated TCN (307k params), 283 anatomy-informed features, local on-device architecture designed for privacy (zero cloud video streaming), and planned pilot study evaluation protocol.

---

## Work-Package Execution Contract

The table below defines the formal acceptance gates across all work packages. Luna and engineering contributors must verify artifacts before marking packages complete:

| Package | Responsible Role | Primary Files & Interfaces | Expected Release Artifact | Acceptance Gate | Downstream Dependencies Unlocked |
|---|---|---|---|---|---|
| **Phase 1: Decoder Calibration** | ML Team | `training/src/metrics.py`<br>`training/src/evaluation.py` | `artifacts/calibrated_decoder_v1/fsm_params.json` | Count MAE $\le 0.40$; Rep End F1 $\ge 60.0\%$ on validation split (test split held locked for final reporting) | Unlocks Phase 2 training and Phase 5 runtime |
| **Phase 2: Fast Runner** | ML Team | `training/src/runner.py`<br>`training/src/data/samplers.py` | `training/configs/experiments/warmstart.yaml` | $4.4\times$ speedup verified; clean checkpoint resumption | Unlocks Phase 3 & 4 fine-tuning runs |
| **Phase 3: Dataset Ingestion** | Data Team | `training/src/data/adapters.py`<br>`docs/dataset-catalog.md` | `data/raw/{dyntherapy, roag, uiprmd}` | Checksums verified; 100% split isolation; 0 participant overlap | Unlocks expanded supervised training |
| **Phase 4: Distillation** | ML Research | `training/src/models/heads.py`<br>`training/src/distill/` | `data/teacher_cache/*.npz` | Distilled student beats supervised baseline on RepCount-pose | Unlocks final student model checkpoint |
| **Phase 5: Mobile Export** | Mobile Team | `training/src/export/`<br>`apps/mobile/` | `artifacts/mobile/adaptfit_tcn_int8.tflite` | 100% parity on 6 golden fixtures; p95 latency $<95\text{ms}$ | Unlocks pilot app deployment |
| **Phase 6: Pilot & Submission** | Product Lead | `docs/pilot-findings.md`<br>`submission/` | Final CAC Video & Application Package | Signed consent; 0 privacy violations; 100% CAC rubric compliance | Final Public Release & Submission |

---

## Decisions that Must Be Resolved

1. **SERE and TULE Institutional Access**: Submit institutional DUA to VisLab Lisbon (`ana.coias@tecnico.ulisboa.pt`) immediately to secure frame-level stroke compensation annotations.
2. **Android Test Device Designation**: Designate a standard test device (e.g. Google Pixel 7 or Samsung Galaxy A54) for all Phase 5 latency and thermal profiling.
3. **Consented Pilot Recruitment**: Partner with local adaptive sports programs and physical therapy clinics to recruit $N=10\text{–}15$ participants across the target profiles.
4. **Offline Teacher Verification**: Verify SSTRAC and PoseRAC inference scripts on the 5 launch exercises before initiating large-scale teacher caching.
