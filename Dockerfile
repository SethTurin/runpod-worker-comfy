# =========================================
# Stage 1: Base image with common dependencies
# =========================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevent prompts from blocking installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels for faster pip installations
ENV PIP_PREFER_BINARY=1
# Disable python output buffering
ENV PYTHONUNBUFFERED=1
# Speed up cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, pip, git, etc.
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install comfy-cli
RUN pip install comfy-cli

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 11.8 --nvidia --version 0.2.7

# Switch working directory to ComfyUI
WORKDIR /comfyui

# Install runpod, requests, face libraries
RUN pip install runpod requests \
    && pip install --use-pep517 facexlib \
    && pip install insightface onnxruntime

# Return to root
WORKDIR /

# Add config & scripts
ADD src/extra_model_paths.yaml /comfyui/
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh /restore_snapshot.sh

# Optionally copy the snapshot file
# (uncomment if you want to include a specific snapshot in your build)
# ADD snapshot*.json /

# Restore the snapshot to install custom nodes
RUN /restore_snapshot.sh

# =========================================
# Stage 2: Downloader
# =========================================
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE

# Switch working directory to ComfyUI
WORKDIR /comfyui

# Install unzip for model extraction
RUN apt-get update && apt-get install -y unzip

# Create directories for base/standard models
RUN mkdir -p models/checkpoints models/vae models/unet models/clip models/controlnet/flux models/pulid models/insightface/models models/upscale_models custom_nodes

# Example CLIP & VAE models that are always downloaded
RUN wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors \
      https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors \
    && wget -O models/vae/ae.safetensors \
      https://aidolonsdata.s3.us-east-1.amazonaws.com/ae.safetensors

# Conditionally download checkpoints, VAE, LoRA, etc.
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -O models/checkpoints/sd_xl_base_1.0.safetensors \
        https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -O models/vae/sdxl_vae.safetensors \
        https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -O models/vae/sdxl-vae-fp16-fix.safetensors \
        https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
    elif [ "$MODEL_TYPE" = "sd3" ]; then \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors \
        https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -O models/unet/flux1-schnell.safetensors \
        https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -O models/clip/clip_l.safetensors \
        https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors \
        https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -O models/vae/ae.safetensors \
        https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        -O models/unet/flux1-dev.safetensors \
        https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
      wget -O models/clip/clip_l.safetensors \
        https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors \
        https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        -O models/vae/ae.safetensors \
        https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
    fi

# -----------------------------------------
# Additional desired downloads
# -----------------------------------------
# UNET (FLUX 1-dev-FP8)
RUN wget -O models/unet/flux1-dev-fp8-e4m3fn.safetensors \
    "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8-e4m3fn.safetensors"

# CLIP
RUN wget -O models/clip/clip_l.safetensors \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
 && wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors \
    "https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/resolve/main/t5xxl_fp8_e4m3fn.safetensors"

# VAE
RUN wget -O models/vae/ae.safetensors \
    "https://aidolonsdata.s3.us-east-1.amazonaws.com/ae.safetensors"

# CONTROLNET
RUN wget -O models/controlnet/flux/flux_controlnet_union.safetensors \
    "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors"

# PULID
RUN wget -O models/pulid/pulid_flux.safetensors \
    "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.0.safetensors"

# INSIGHTFACE
RUN wget -O models/insightface/models/antelopev2.zip \
    "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip" \
 && unzip models/insightface/models/antelopev2.zip -d models/insightface/models/

# Upscale models
RUN wget -O models/upscale_models/4x-ClearRealityV1.pth \
    "https://aidolonsdata.s3.us-east-1.amazonaws.com/4x-ClearRealityV1.pth"

# Download and unzip custom nodes
RUN wget -O custom_nodes.zip \
    "https://aidolonsdata.s3.us-east-1.amazonaws.com/custom_nodes.zip" \
 && unzip -o custom_nodes.zip -d /comfyui/custom_nodes

# =========================================
# Stage 3: Final Image
# =========================================
FROM base AS final

# Copy models from the downloader stage
COPY --from=downloader /comfyui/models /comfyui/models
COPY --from=downloader /comfyui/custom_nodes /comfyui/custom_nodes

# Expose working directory if needed
WORKDIR /comfyui

# Final command
CMD ["/start.sh"]
