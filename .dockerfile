FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Update package lists and install git-lfs
RUN apt update && \
    apt-get install -y git-lfs && \
    git lfs install

# Install specific versions of PyTorch, torchvision, and torchaudio
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy the requirements file and install additional Python packages
COPY . .
RUN pip install -r requirements.txt
