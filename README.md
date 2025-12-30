# ComfyUI Segment Anything (A100 Ultra-Optimized Edition) 🚀

This project is a heavily optimized fork of the original ComfyUI Segment Anything, specifically engineered for **NVIDIA A100 (80GB)** GPUs. It pushes the hardware to its absolute limit using BF16 precision, VRAM Locking, and Zero-Copy GPU processing.

> **Note:** While this version works best on A100, it can technically run on other high-VRAM Ampere/Hopper GPUs (RTX 3090/4090/H100), but settings like VRAM Lock need to be adjusted accordingly.

## ⚡ Key Features (A100 Exclusive)

1.  **God Speed Mode:** - Replaced CPU resizing with **GPU-Native Interpolation**.
    - Processing pipeline stays 100% on VRAM (Zero CPU Offload).
    - Capable of processing massive batches (128+ frames) in seconds.
2.  **All-in-One Architecture:**
    - Combined System Patcher, VRAM Optimizer, and Inference into a **single node**. No need for complex wiring.
3.  **Auto-Patching System:**
    - Automatically hot-patches PyTorch runtime errors (`meshgrid` indexing warnings, `checkpoint` reentrant errors) without modifying system files.
4.  **VRAM Stabilizer:**
    - Includes a `VRAM_Lock_GB` feature to reserve specific memory blocks, preventing garbage collection spikes and ensuring "flat-line" memory usage.
5.  **Precision Engineering:**
    - **SAM**: Runs in `BFloat16` (BF16) for maximum throughput on Tensor Cores.
    - **GroundingDino**: Runs in `Float32` to ensure text encoder stability.
    - **TF32**: Enabled globally for matrix multiplications.

## 🛠️ Requirements

Ensure you have the standard ComfyUI dependencies installed.
```bash
pip3 install -r requirements.txt