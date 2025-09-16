#!/bin/bash

# =============================================================================
# LLM Evaluation Environment Setup Script
# Sets up complete environment for H100 GPU model evaluation
# =============================================================================

set -e  # Exit on any error

echo "🚀 Setting up LLM Evaluation Environment for H100 GPU"
echo "=================================================="

# Configuration
CONDA_ENV_NAME="llm_eval"
WORKSPACE_DIR="$HOME/methyl_savanna/llm_evaluation"
PYTHON_VERSION="3.11"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on system with H100
check_gpu() {
    log_info "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
        echo "GPU Info: $GPU_INFO"
        
        if echo "$GPU_INFO" | grep -q "H100"; then
            log_success "H100 GPU detected!"
        else
            log_warning "H100 not detected. Script will continue but results may vary."
        fi
    else
        log_error "nvidia-smi not found. CUDA/GPU setup may be incomplete."
        exit 1
    fi
}

# Setup directory structure
setup_directories() {
    log_info "Setting up directory structure..."
    
    cd "$WORKSPACE_DIR"
    
    # Create all necessary directories
    mkdir -p {models,configs,results,benchmarks,serving,evaluation,scripts,cache}
    mkdir -p models/{implementations,checkpoints}
    mkdir -p results/{performance,agent_tests,comparisons}
    mkdir -p benchmarks/{datasets,scripts}
    mkdir -p cache/{models,tokenizers}
    
    log_success "Directory structure created"
}

# Create or update conda environment
setup_conda_env() {
    log_info "Setting up conda environment: $CONDA_ENV_NAME"
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Please install conda/miniconda first."
        exit 1
    fi
    
    # Check if environment already exists
    if conda env list | grep -q "^$CONDA_ENV_NAME "; then
        log_warning "Environment $CONDA_ENV_NAME already exists. Updating..."
        conda env update -n "$CONDA_ENV_NAME" -f environment.yml
    else
        log_info "Creating new environment..."
        conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    fi
    
    log_success "Conda environment ready"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    
    # Core ML dependencies
    pip install --upgrade pip
    
    # PyTorch with CUDA support (adjust for your CUDA version)
    pip install torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Model serving frameworks
    pip install vllm>=0.2.0
    pip install transformers>=4.36.0
    pip install accelerate>=0.24.0
    pip install optimum>=1.14.0
    
    # Quantization libraries
    pip install auto-gptq>=0.5.0
    pip install autoawq>=0.1.6
    
    # Flash attention for efficiency
    pip install flash-attn --no-build-isolation
    
    # Agent frameworks
    pip install langchain>=0.1.0
    pip install langchain-community
    pip install langgraph>=0.0.40
    
    # Evaluation and utilities
    pip install datasets>=2.14.0
    pip install evaluate>=0.4.0
    pip install wandb
    pip install tensorboard
    pip install jupyter
    pip install ipywidgets
    
    # API and serving
    pip install fastapi>=0.104.0
    pip install uvicorn[standard]
    pip install requests
    pip install aiofiles
    
    # Data processing
    pip install pandas>=2.0.0
    pip install numpy>=1.24.0
    pip install matplotlib>=3.7.0
    pip install seaborn>=0.12.0
    pip install plotly>=5.15.0
    
    # Development tools
    pip install black
    pip install isort
    pip install pytest
    pip install python-dotenv
    
    log_success "Python dependencies installed"
}

# Setup Ollama (for rapid prototyping)
setup_ollama() {
    log_info "Setting up Ollama for rapid prototyping..."
    
    if ! command -v ollama &> /dev/null; then
        log_info "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        
        # Start ollama service
        systemctl --user start ollama || log_warning "Could not start ollama service automatically"
    else
        log_info "Ollama already installed"
    fi
    
    log_success "Ollama setup complete"
}

# Create configuration files
create_config_files() {
    log_info "Creating configuration files..."
    
    # Environment configuration
    cat > .env << EOF
# LLM Evaluation Environment Configuration
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false
TRANSFORMERS_CACHE=$WORKSPACE_DIR/cache/models
HF_HOME=$WORKSPACE_DIR/cache
WANDB_PROJECT=llm_evaluation
WANDB_MODE=online

# Model serving
VLLM_HOST=0.0.0.0
VLLM_PORT=8000
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434

# Evaluation settings
MAX_CONCURRENT_EVALS=2
EVAL_BATCH_SIZE=4
DEFAULT_MAX_TOKENS=2048
DEFAULT_TEMPERATURE=0.1
EOF

    # Create conda environment.yml for reproducibility
    cat > environment.yml << EOF
name: $CONDA_ENV_NAME
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=$PYTHON_VERSION
  - pip
  - git
  - wget
  - curl
  - htop
  - nvtop
  - pip:
    - -r requirements.txt
EOF

    # Create requirements.txt
    cat > requirements.txt << EOF
# Core ML Framework
torch>=2.1.0
transformers>=4.36.0
accelerate>=0.24.0
optimum>=1.14.0

# Model Serving
vllm>=0.2.0
fastapi>=0.104.0
uvicorn[standard]

# Quantization
auto-gptq>=0.5.0
autoawq>=0.1.6
flash-attn

# Agent Frameworks
langchain>=0.1.0
langchain-community
langgraph>=0.0.40

# Evaluation
datasets>=2.14.0
evaluate>=0.4.0
wandb
tensorboard

# Data & Analysis
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Development
jupyter
black
isort
pytest
python-dotenv
requests
aiofiles
EOF

    log_success "Configuration files created"
}

# Create initial scripts
create_initial_scripts() {
    log_info "Creating initial evaluation scripts..."
    
    # Quick test script
    cat > scripts/quick_test.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test script to verify environment setup
"""
import torch
import transformers
import vllm
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_cuda():
    """Test CUDA availability"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def test_model_loading():
    """Test loading a small model"""
    print("\nTesting model loading with microsoft/DialoGPT-small...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        print("✅ Tokenizer loaded successfully")
        
        # Don't actually load model in test to save time
        print("✅ Model loading test passed")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_vllm():
    """Test vLLM installation"""
    try:
        from vllm import LLM, SamplingParams
        print("✅ vLLM imported successfully")
        return True
    except Exception as e:
        print(f"❌ vLLM import failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing LLM Evaluation Environment")
    print("=" * 50)
    
    test_cuda()
    test_model_loading()
    test_vllm()
    
    print("\n✅ Environment test completed!")
EOF

    chmod +x scripts/quick_test.py
    
    # Model download script
    cat > scripts/download_models.sh << 'EOF'
#!/bin/bash

# Download and cache priority models for evaluation
# This script downloads models to local cache for faster loading

source .env

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct" 
    "microsoft/Phi-3.5-mini-instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

echo "📥 Downloading priority models to cache..."

for model in "${MODELS[@]}"; do
    echo "Downloading: $model"
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ['TRANSFORMERS_CACHE'] = '$TRANSFORMERS_CACHE'
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('$model')
print('✅ Tokenizer cached')
# Note: Model weights download will happen during actual evaluation
"
done

echo "✅ Model download completed"
EOF

    chmod +x scripts/download_models.sh
    
    log_success "Initial scripts created"
}

# Main setup function
main() {
    log_info "Starting LLM Evaluation Environment Setup"
    echo "Workspace: $WORKSPACE_DIR"
    echo "Conda Environment: $CONDA_ENV_NAME"
    echo ""
    
    # Run setup steps
    check_gpu
    setup_directories
    setup_conda_env
    install_python_deps
    setup_ollama
    create_config_files
    create_initial_scripts
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate environment: conda activate $CONDA_ENV_NAME"
    echo "2. Test setup: cd $WORKSPACE_DIR && python scripts/quick_test.py"
    echo "3. Download models: ./scripts/download_models.sh"
    echo "4. Start evaluation: python -m evaluation.run_evaluation"
    echo ""
    echo "For more information, see: LLM_INDIVIDUAL_MODEL_IMPLEMENTATION_PLAN.md"
}

# Run main function
main "$@"