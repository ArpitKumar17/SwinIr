# SwinIR - Image Super-Resolution on the Stanford Cars Dataset
# Overview
This project focuses on enhancing low-resolution images using the SwinIR architecture. We utilized the Stanford Cars Dataset from Kaggle, which contains 16,185 images across 196 car classes. Our model aims to restore fine details in blurry and distorted images, significantly improving image quality.

# Training Details
-> Dataset: Stanford Cars Dataset (Kaggle)

-> Training Time: ~2 hours 45 minutes

-> Hardware: NVIDIA T4 GPU (Google Colab)

# Methodology
We employed SwinIR, a transformer-based image restoration model, which leverages a self-attention mechanism to capture:

-> Low-level textures and patterns

-> High-level structural information

# Results & Challenges
Despite hardware limitations, our model successfully enhanced image clarity and restored minute details. However, with access to more powerful GPUs, the performance could be further improved.

# Conclusion
This project demonstrates the potential of SwinIR for image super-resolution, even under computational constraints. We hope our work contributes to further advancements in image enhancement.
