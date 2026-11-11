# Deep Dive: Systems-Level Real-Time Telemetry & Hardware Benchmarking – Webcam AI-Upscaling Pipelines

## 1. Motivation & Problem Statement
Modern computer vision applications, real-time video streaming, and edge AI workloads frequently hit severe hardware bottlenecks when scaling beyond standard definitions. Consumer-grade webcams are frequently limited by native resolution constraints, heavy compression artifacts, or extreme CPU overhead during raw frame acquisition, colour-space conversion (`BGR` to `RGB`), and pipeline orchestration. 

This project explores the performance, latency characteristics, and hardware footprint of scaling a live video streaming pipeline from a naive CPU-bound baseline to fully hardware-accelerated deep learning architectures (including custom simulated upscalers and true Sub-Pixel Convolution via ESPCN). Operating on a constrained consumer system—an NVIDIA RTX 3050 (6 GB VRAM) paired with an Intel Core i3-12100F—this benchmark investigates how software abstractions, memory transfers, and neural network inference impact real-time frame rates, thermal stability, and frame-pacing consistency.

---

## 2. Theoretical Background: The Anatomy of Real-Time Inference Pipelines
Real-time video processing differs fundamentally from batch inference because it operates under strict temporal boundaries. At a target framerate of 30 frames per second (FPS), a single frame must be captured, preprocessed, inferred, and rendered within a rigid **33.3 ms latency budget**.

* **Data Layout & Memory Permutations:** Moving from OpenCV’s native `HWC` (Height, Width, Channels) BGR memory layout to PyTorch’s `BCHW` (Batch, Channels, Height, Width) RGB tensor requires continuous memory copies, non-contiguous memory slicing, and normalization (`/ 255.0`). These operations introduce hidden CPU-GPU synchronization overheads if not handled asynchronously.
* **The Inference & Reconstruction Challenge:** While simple bilinear interpolation is computationally trivial, neural network-based upscaling introduces dense convolutional layers and non-linear activations (`ReLU`). Without architectural safeguards like residual connections or sub-pixel shuffling (`nn.PixelShuffle`), un-trained or poorly structured models easily swamp memory bandwidth, cause severe color distortion, or blow past the real-time latency budget.

---

## 3. Test Environment
* **CPU:** Intel Core i3-12100F (4 cores / 8 threads)
* **System RAM:** 16 GB DDR4
* **GPU:** NVIDIA GeForce RTX 3050 — 6 GB VRAM
* **Software Stack:** Ubuntu 22.04 LTS, Python 3.11.15, PyTorch (CUDA backend), OpenCV, Pandas, Matplotlib, Pynvml, Psutil

---

## 4. Deep-Dive Insight: The Pitfall of Un-Trained Weights & Channel Distortion
The experiment is structured as a progressive four-phase benchmark designed to evaluate system performance and telemetry dynamics as video processing transitions from a CPU baseline to hardware-accelerated deep learning architectures. During coding, a critical engineering hurdle encountered during the implementation of Phases 3 and 4 highlights a fundamental gap between theoretical deep learning and practical execution: **the behavior of un-trained neural networks with random initialization**.

### The Phenomenon: Psychedelic Color Shifts
When passing raw video frames through custom convolutional layers (`Conv2d`) initialized without pre-trained weights, the resulting output frames initially exhibited severe color inversions, flickering artifacts, and unnatural dominant hues (such as neon green, turquoise, or deep purple). 

### Why It Happens
Unlike standard image processing filters (where matrix kernels are mathematically hardcoded for specific tasks like edge detection or blurring), an un-trained CNN consists entirely of randomly distributed weights. 
1. **Arbitrary Channel Mixing:** When 3-channel (RGB) inputs pass through consecutive layers of random floating-point weights, the network arbitrarily multiplies, scales, and cross-contaminates the red, green, and blue channels without semantic awareness of actual color spaces.
2. **Activation Clipping:** Non-linear activation functions like `ReLU` combined with unscaled random weights frequently push pixel values past valid boundaries, causing extreme clipping when normalized back to `[0, 255]`.

### The Engineering Solution: Residual Connections & Identity Initialization
To tame the network without requiring a heavy training phase, two architectural safeguards were introduced:
* **Residual Connections (`out + baseline`):** Instead of forcing the CNN to reconstruct the entire image from scratch, the network is forced to learn only a high-frequency *residual map* (details, edges, and sharpening) while the original bilinear upscaled frame acts as a stable background baseline.
* **Gentle Weight Initialization:** Initializing the final convolutional layer weights extremely close to zero (`std=0.001`) ensures that the network starts its execution loop in a nearly transparent state, preserving the natural color distribution of the live webcam feed from the very first frame.

---

## 5. The Experimental Journey & Practical Observations ("What I Experienced")
During the 4-phase benchmark progression, distinct performance dynamics became apparent:

1. **Phase 1 (Baseline CPU):** Processing frames entirely on the CPU yields a nominal framerate (~31 FPS), but suffers from high temporal instability and micro-stutters (FPS Jitter > 11 FPS) as background operating system threads interrupt the sequential execution queue.

<img width="3600" height="2400" alt="phase1_detailed_report" src="https://github.com/user-attachments/assets/8d18feef-117f-46ab-91d4-07b656a5733e" />

2. **Phase 2 (GPU Transfer):** Offloading tensor creation and memory allocation to the GPU reduces CPU core strain, but introduces initial Host-to-Device (H2D) memory copy overheads and minor latency spikes during synchronization checkpoints.

<img width="3600" height="2400" alt="phase2_detailed_report" src="https://github.com/user-attachments/assets/7b94c10d-0963-4793-86a3-71635b56a0fe" />

3. **Phase 3 (AI-Upscaling with Residuals):** Introducing a 3-layer convolutional upscaler with a bilinear residual connection successfully preserves natural color fidelity while shifting computational load to parallel GPU cores. Framerates stabilize significantly (Jitter drops to ~1.38 FPS), though average FPS dips to ~25.18 due to deep inference overhead.

<img width="3600" height="2400" alt="phase3_detailed_report" src="https://github.com/user-attachments/assets/d35f2ffa-a788-4e49-890f-bc7afea2e4e3" />

4. **Phase 4 (ESPCN Architecture):** Implementing a true Efficient Sub-Pixel Convolutional Network (`nn.PixelShuffle`) shifts upscaling math into deep tensor channels. While memory footprint remains well-controlled (~52 MB VRAM), the heavier computational graph pushes inference latency to ~30.99 ms, pushing right against the limits of the 33 ms frame budget at ~19.84 FPS.

<img width="3600" height="2400" alt="phase4_espcn_detailed_report" src="https://github.com/user-attachments/assets/ad141bc4-0a37-49d1-9487-3bbd95cc1745" />

---

## 6. Comprehensive Benchmark Results

| Phase | Avg FPS | FPS StdDev (Jitter) | 5% Lows FPS | Lat Ø (ms) | Lat Max (ms) | Peak VRAM (MB) | Peak RAM (MB) | Max Temp (°C) | GPU Load (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 (Baseline CPU)** | 31.19 | 11.13 | 24.96 | 0.00 | 0.00 | 0.00 | 867.25 | 33°C | 24.2% |
| **Phase 2 (GPU Transfer)** | 31.19 | 9.76 | 24.87 | 0.51 | 79.61 | 5.27 | 990.53 | 35°C | 5.3% |
| **Phase 3 (AI-Upscaling)** | 25.18 | 1.38 | 23.59 | 19.61 | 142.99 | 63.32 | 1514.34 | 48°C | 54.8% |
| **Phase 4 (ESPCN AI-Upscaling)** | 19.84 | 1.02 | 19.40 | 30.99 | 148.22 | 52.84 | 1344.81 | 56°C | 65.2% |


<img width="4800" height="3000" alt="comprehensive_benchmark_analysis" src="https://github.com/user-attachments/assets/4f5abca5-b297-4836-bf77-46818bc79f66" />


---

## 7. Real-World Implications: Why Systems-Level Telemetry Matters

Moving from abstract high-level programming to systems-level performance engineering reveals critical dynamics that standard profilers often miss:

* **The Frame-Pacing vs. Raw Throughput Paradox:** 
  A fascinating observation in the telemetry data is that while Phase 3 and Phase 4 exhibit lower *average* framerates than the CPU baseline, their **temporal stability (FPS StdDev / Jitter)** improves dramatically (dropping from 11.13 down to ~1.02). On the CPU, background OS tasks cause unpredictable micro-stutters. On the GPU, the hardware command queue acts as a synchronous buffer, smoothing out execution jitter and delivering a much more consistent, predictable visual stream.
* **The 33ms Latency Wall & Headroom Analysis:** 
  In real-time rendering and video pipelines, average latency can be deceptive. Phase 4 reports an average inference latency of ~30.99 ms, which appears safe for a 30 FPS target. However, looking at the **Max Latency spikes (up to 148 ms)** reveals occasional pipeline stalls caused by Python garbage collection, OpenCV buffer locks, or CUDA context switches. In production environments, these maximum spikes cause visible frame drops and stuttering.
* **Memory Bandwidth & PCIe Bottlenecks:** 
  Continuous streaming from CPU memory to GPU VRAM via PCIe lanes introduces subtle overheads. Even though VRAM consumption remains modest (~52–63 MB), the frequency of tensor allocation and deletion stresses the Python memory allocator, highlighting the need for pre-allocated memory pools in production-grade systems engineering.
* **Thermal & Power Scaling in Constrained Hardware:** 
  As workloads shift to deep learning architectures, GPU core utilization scales linearly from 5% up to 65.2%, driving core temperatures from 33°C to 56°C. While 56°C is well within safe operating limits for an RTX 3050, it demonstrates that running local edge AI tasks places sustained thermal stress on hardware cooling systems, which directly impacts boost clocks during extended operation.

---

## 8. Conclusion & Future Work

This project demonstrates that moving from CPU-bound video processing to GPU-accelerated deep learning pipelines requires careful balance between algorithmic complexity and strict real-time latency budgets. Even on entry-level hardware like the RTX 3050, structuring tensors correctly and utilizing architectures like ESPCN unlocks viable real-time AI capabilities, mirroring the optimization principles found in modern gaming upscalers (DLSS/FSR).

### Future Work & Advanced Enterprise Metrics
To elevate this benchmark to a professional enterprise level (HPC / Production Edge AI), future iterations will incorporate the following advanced telemetry metrics:

1. **Power Consumption (Wattage):**
   * *Tracking:* Queryable via NVML (`nvmlDeviceGetPowerUsage`).
   * *Benefit:* Determines energy efficiency (FPS per watt). Shows whether the performance gain from the GPU justifies the additional power draw.
2. **PCIe Bandwidth / Transfer Latency (Host-to-Device Copy Time):**
   * *Tracking:* Measuring the exact duration of memory transfer from CPU RAM to GPU VRAM (`torch.cuda.synchronize()` before and after `.to(device)`).
   * *Benefit:* Identifies whether the PCIe bus becomes a bottleneck at higher resolutions (e.g., 4K).
3. **1% and 0.1% Lows (Frametime Percentiles):**
   * *Tracking:* Statistical evaluation of extreme frametimes instead of pure average FPS.
   * *Benefit:* The industry standard in gaming and media to objectively quantify stuttering and micro-stutters.
4. **Thermal Throttling Indicator:**
   * *Tracking:* Querying `nvmlDeviceGetPerformanceState` or monitoring clock frequencies (`nvmlDeviceGetClockInfo`).
   * *Benefit:* Shows whether longer test runs (e.g., > 10 minutes) lead to clock speed reductions due to thermal buildup.

Additionally, near-term optimization goals include:
* **TensorRT / ONNX Runtime Export:** Bypassing the Python interpreter overhead by compiling the PyTorch model to TensorRT to push inference latency well below 10 ms.
* **Zero-Copy Memory Interop:** Integrating direct CUDA-OpenGL/Vulkan interoperability to eliminate CPU-side numpy/opencv staging buffers entirely.
