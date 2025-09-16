#!/bin/bash
# LLM Evaluation Environment Setup Script
# For Python 3.12.8 on H100 GPU system

set -e  # Exit on any error

echo "🚀 Setting up LLM Evaluation Environment..."
module load python3/2025.1-py312
echo "Python version: $(crun -p ~/envs/llm_env python --version)"
echo "Working directory: $(pwd)"

# Upgrade pip first
echo "📦 Upgrading pip..."
crun -p ~/envs/llm_env pip install --upgrade pip setuptools wheel

# Install core packages first (without CUDA)
echo "🔧 Installing core packages..."
crun -p ~/envs/llm_env pip install --upgrade \
    numpy \
    setuptools-scm \
    packaging \
    wheel \
    cython

# Install PyTorch with CUDA support - Latest stable version
echo "🔥 Installing PyTorch with CUDA 12.1 support (latest stable)..."
crun -p ~/envs/llm_env pip install torch>=2.4.0 torchvision>=0.19.0 torchaudio>=2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install transformers and related packages - Latest compatible versions
echo "🤖 Installing Transformers ecosystem..."
crun -p ~/envs/llm_env pip install \
    "transformers>=4.44.0" \
    "tokenizers>=0.19.0" \
    "accelerate>=0.33.0" \
    "safetensors>=0.4.0" \
    "huggingface-hub>=0.24.0"

# Install model serving libraries - Latest compatible versions
echo "⚡ Installing model serving libraries..."
crun -p ~/envs/llm_env pip install "vllm>=0.6.0"

# Install flash-attention (may take time to compile)
echo "⚡ Installing flash-attention (this may take several minutes)..."
crun -p ~/envs/llm_env pip install "flash-attn>=2.6.0" --no-build-isolation

# Install quantization libraries - Skip auto-gptq due to compilation issues
echo "🗜️ Installing quantization libraries..."
crun -p ~/envs/llm_env pip install \
    "bitsandbytes>=0.43.0" \
    "optimum>=1.21.0"
echo "⚠️  Skipping auto-gptq due to CUDA compilation issues - can be added later if needed"

# Install remaining packages from fixed requirements
echo "📋 Installing remaining packages from requirements-fixed.txt..."
crun -p ~/envs/llm_env pip install -r requirements-fixed.txt

# Verify installation
echo "✅ Verifying installation..."
crun -p ~/envs/llm_env python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
crun -p ~/envs/llm_env python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
crun -p ~/envs/llm_env python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Check if we can import key libraries
echo "🔍 Testing key library imports..."
crun -p ~/envs/llm_env python -c "
try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
    import vllm
    print(f'✅ vLLM: {vllm.__version__}')
    import flash_attn
    print(f'✅ Flash Attention: {flash_attn.__version__}')
    print('✅ All key libraries imported successfully!')
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('⚠️  Some libraries may need additional setup')
"

echo "🎉 Environment setup complete!"
echo "💡 To activate: crun -p ~/envs/llm_env"
echo "📊 Check GPU memory: nvidia-smi"