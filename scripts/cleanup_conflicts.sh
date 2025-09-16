#!/bin/bash
# Cleanup and Fix Version Conflicts Script
# Run this before the main installation to resolve conflicts

set -e

echo "🧹 Cleaning up conflicting packages..."
module load python3/2025.1-py312

# Remove conflicting packages
echo "🗑️  Uninstalling conflicting versions..."
crun -p ~/envs/llm_env pip uninstall -y torch torchvision torchaudio vllm transformers || true
crun -p ~/envs/llm_env pip uninstall -y xformers flash-attn auto-gptq || true
crun -p ~/envs/llm_env pip uninstall -y numpy scipy || true

# Fix setuptools version
echo "🔧 Fixing setuptools version..."
crun -p ~/envs/llm_env pip install "setuptools>=77.0.3,<80"

# Install compatible numpy first to avoid conflicts
echo "📊 Installing compatible numpy..."
crun -p ~/envs/llm_env pip install "numpy>=1.24.0,<2.1.0"

echo "✅ Cleanup complete! Ready for fresh installation."
echo "💡 Now run: ./install_packages.sh"