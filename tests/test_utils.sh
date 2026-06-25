#!/bin/bash
# ============================================================================
# RFDETRs-CPP Test Utilities
# Shared functions sourced by all task test scripts.
# ============================================================================

# Colors
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export NC='\033[0m'

# Virtual environment directory (shared across all task test runs)
VENV_DIR="${TEST_VENV_DIR:-$HOME/.rfdetrs-cpp-test-venv}"

# ============================================================================
# Output helpers
# ============================================================================
print_header() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# ============================================================================
# Install uv (fast Python package installer)
# ============================================================================
install_uv() {
    if command -v uv &>/dev/null; then
        echo -e "${GREEN}uv already installed${NC}"
        return 0
    fi

    echo -e "${YELLOW}Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if command -v uv &>/dev/null; then
        echo -e "${GREEN}uv installed${NC}"
        return 0
    fi
    echo -e "${RED}Failed to install uv${NC}"
    return 1
}

# ============================================================================
# Setup virtual environment
# ============================================================================
setup_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        echo -e "${BLUE}Creating venv at $VENV_DIR${NC}"
        if command -v uv &>/dev/null; then
            uv venv "$VENV_DIR"
        else
            python3 -m venv "$VENV_DIR"
        fi
    fi
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    echo -e "${GREEN}Virtual environment activated${NC}"
}

# ============================================================================
# Install Python packages (uv preferred, pip fallback)
# ============================================================================
install_python_packages() {
    local packages="$*"
    echo -e "${BLUE}Installing Python packages: $packages${NC}"
    setup_venv
    if command -v uv &>/dev/null; then
        uv pip install $packages --quiet && return 0
    fi
    pip install -q $packages && return 0
    echo -e "${RED}Failed to install packages${NC}"
    return 1
}

# ============================================================================
# Export an RF-DETR model with random weights to a target .onnx path.
#
# Usage: export_rfdetr_model <ClassName> <output_path>
# Example: export_rfdetr_model RFDETRNano models/rfdetr-nano.onnx
# ============================================================================
export_rfdetr_model() {
    local class_name="$1"
    local output_path="$2"

    echo -e "${BLUE}Exporting ${class_name} (random weights) → ${output_path}${NC}"

    python3 - "$class_name" "$output_path" <<'PYEOF'
import sys, shutil, tempfile, glob
import rfdetr as _pkg
cls_name, out_path = sys.argv[1], sys.argv[2]
cls = getattr(_pkg, cls_name, None)
if cls is None:
    raise AttributeError(f'{cls_name} not found in rfdetr package')
m = cls(pretrain_weights=None)
with tempfile.TemporaryDirectory() as tmp:
    m.export(output_dir=tmp)
    srcs = glob.glob(tmp + '/**/*.onnx', recursive=True)
    if not srcs:
        raise FileNotFoundError(f'No .onnx produced by {cls_name}.export()')
    shutil.copy2(srcs[0], out_path)
    print(f'Exported: {out_path}')
PYEOF
    print_success "Model exported: ${output_path}"
}
