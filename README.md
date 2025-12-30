# ComfyUI Segment Anything (A100 Ultra-Optimized Edition) 🚀

This project is a heavily optimized fork of the original ComfyUI Segment Anything, specifically engineered for **NVIDIA A100 (80GB)** GPUs. It pushes the hardware to its absolute limit using BF16 precision, VRAM Locking, and Zero-Copy GPU processing.

> **Note:** While this version works best on A100, it can technically run on other high-VRAM Ampere/Hopper GPUs (RTX 3090/4090/H100), but settings like VRAM Lock need to be adjusted accordingly.

## ⚡ Key Features (v0.9.1)

1.  **God Speed Mode:** - Replaced CPU resizing with **GPU-Native Interpolation**.
    - Processing pipeline stays 100% on VRAM (Zero CPU Offload).
    - Capable of processing massive batches (128+ frames) in seconds.
    
2.  **Streaming Pipeline (Stable VRAM):**
    - **No more OOM!** The node intelligently chunks input frames from CPU to GPU.
    - VRAM usage remains flat and stable regardless of video length (100 or 1000 frames).
    
3.  **Direct Masked Output:**
    - The `IMAGE` output now automatically returns the object with a **black background**.
    - No need for extra `ImageCompositeMasked` nodes.
    
4.  **All-in-One Architecture:**
    - Combined System Patcher, VRAM Optimizer, and Inference into a **single node**. No need for complex wiring.

5.  **Auto-Patching System:**
    - Automatically hot-patches PyTorch runtime errors (`meshgrid` indexing warnings, `checkpoint` reentrant errors) without modifying system files.

6.  **VRAM Stabilizer:**
    - Includes a `VRAM_Lock_GB` feature to reserve specific memory blocks, preventing garbage collection spikes.
    - `sam_batch_size` slider is optimized (max 128) for precise UI control.

7.  **Precision Engineering:**
    - **SAM**: Runs in `BFloat16` (BF16) for maximum throughput on Tensor Cores.
    - **GroundingDino**: Optimized batch inference.

## 🛠️ Installation

1. Clone this repository into your `ComfyUI/custom_nodes/` folder:
    ```bash
    cd ComfyUI/custom_nodes/
    git clone https://github.com/huyl3-cpu/comfyui_segment_anything_a100.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    > **Note:** My test with:
    > ```bash
    > pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
    > ```
    > on Google Colab A100.

3. Download Models:
    - The node will automatically download necessary models (`GroundingDINO_SwinT_OGC`, `sam_vit_b`, etc.) to `ComfyUI/models/` on the first run.

## 💻 Usage

1.  **Add Node:** Right-click -> `segment_anything` -> `GroundingDinoSAM Segment (v0.9.1)`.
2.  **Connect:**
    - Input `IMAGE`: Connect your video or image batch.
    - Input `SAM_MODEL` & `GROUNDING_DINO_MODEL`: Load via respective loader nodes.
3.  **Settings:**
    - `prompt`: Enter the text description of the object (e.g., "black shirt", "person").
    - `sam_batch_size`: Set to **32** or **64** for the best balance between speed and VRAM stability.
    - `A100_Optim`: Enable for GPU acceleration.

## 🤝 Contribution

If you encounter any errors or have suggestions for optimization, please feel free to contribute, open an issue, or submit a Pull Request to help improve this project.

*Happy Segmenting!*