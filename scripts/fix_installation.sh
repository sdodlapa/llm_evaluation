#!/bin/bash
# Fix version conflicts and install remaining packages

echo "🔧 Fixing version conflicts..."

# Fix setuptools version for vLLM compatibility
crun -p ~/envs/llm_env pip install "setuptools>=77.0.3,<80"

# Install compatible PyTorch version for vLLM
echo "🔥 Installing compatible PyTorch for vLLM..."
crun -p ~/envs/llm_env pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
echo "⚡ Installing vLLM..."
crun -p ~/envs/llm_env pip install vllm

# Install flash-attention with specific CUDA flags
echo "⚡ Installing flash-attention..."
crun -p ~/envs/llm_env pip install flash-attn --no-build-isolation

# Skip auto-gptq for now due to compilation issues, install alternatives
echo "🗜️ Installing quantization libraries (skipping auto-gptq)..."
crun -p ~/envs/llm_env pip install bitsandbytes optimum

# Install transformers ecosystem
echo "🤖 Installing transformers..."
crun -p ~/envs/llm_env pip install transformers tokenizers accelerate safetensors huggingface-hub

# Install basic evaluation packages
echo "📊 Installing evaluation packages..."
crun -p ~/envs/llm_env pip install datasets evaluate numpy pandas matplotlib seaborn tqdm rich

echo "✅ Fixed installation complete!"