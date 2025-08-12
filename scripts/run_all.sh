#!/bin/bash

# Capacity-Time Dilation Project: Run All Tests and Experiments
# This script sets up the environment, runs tests, and executes all notebooks

set -e  # Exit on any error

echo "=============================================="
echo "Capacity-Time Dilation: Full Test Suite"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Setup environment
echo ""
echo "1. Setting up environment..."

# Check if Python 3.11+ is available
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 11 ]); then
    echo "Error: Python 3.11+ required. Found: $python_version"
    exit 1
fi

echo "✓ Python version: $python_version"

# Install dependencies
echo "Installing dependencies..."
pip install -r env/requirements.txt

echo "✓ Dependencies installed"

# Run tests
echo ""
echo "2. Running unit tests..."
echo "========================"

# Run pytest with verbose output
python -m pytest tests/ -v --tb=short

if [ $? -eq 0 ]; then
    echo "✓ All tests passed"
else
    echo "✗ Some tests failed"
    echo "Check test output above for details"
    exit 1
fi

# Create figures directory
mkdir -p figures

# Run notebooks
echo ""
echo "3. Executing Jupyter notebooks..."
echo "=================================="

# Check if jupyter is available
if ! command -v jupyter &> /dev/null; then
    echo "Installing jupyter..."
    pip install jupyter nbconvert
fi

# List of notebooks to run
notebooks=(
    "notebooks/01_KS1_clock_quadratic.ipynb"
    "notebooks/02_KS2_motion_vs_demand.ipynb" 
    "notebooks/03_KS3_backreaction_toy.ipynb"
    "notebooks/04_KS4_response_kernel.ipynb"
)

# Execute each notebook
for notebook in "${notebooks[@]}"; do
    echo ""
    echo "Running: $notebook"
    echo "----------------------------------------"
    
    # Execute notebook and save output
    jupyter nbconvert --to notebook --execute --inplace "$notebook"
    
    if [ $? -eq 0 ]; then
        echo "✓ $notebook completed successfully"
    else
        echo "✗ $notebook failed"
        echo "Check notebook for error details"
        exit 1
    fi
done

# Export figures
echo ""
echo "4. Exporting figures..."
echo "======================"

python scripts/export_figures.py

if [ $? -eq 0 ]; then
    echo "✓ Figures exported"
else
    echo "✗ Figure export failed"
fi

# Summary
echo ""
echo "=============================================="
echo "FULL SUITE COMPLETED SUCCESSFULLY"
echo "=============================================="

echo ""
echo "Results:"
echo "  - Unit tests: PASSED"
echo "  - KS-1 (Clock Quadratic): Check notebook output"
echo "  - KS-2 (Motion vs Demand): Check notebook output" 
echo "  - KS-3 (Back-Reaction): Check notebook output"
echo "  - KS-4 (Response Kernel): Check notebook output"
echo ""
echo "Generated files:"
echo "  - Figures: figures/*.png"
echo "  - Updated notebooks with output cells"
echo ""
echo "Next steps:"
echo "  1. Review notebook outputs for KS-1 through KS-4 results"
echo "  2. Check figures/ directory for plots"
echo "  3. Update docs/results_log.md with findings"
echo ""

# Optional: Display quick results summary if available
if [ -f "figures/experiment_summary.txt" ]; then
    echo "Experiment Summary:"
    echo "=================="
    cat figures/experiment_summary.txt
fi

echo "Run complete!"
