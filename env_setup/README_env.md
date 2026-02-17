# Environment Setup

This guide provides instructions for setting up the `kao310` conda environment on Ubuntu 22.04.

---

## 1. Download and Install Miniconda
```bash
# Download the Miniconda installer (Python 3.10 version)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run the installation script
bash Miniconda3-latest-Linux-x86_64.sh

# Follow the prompts:
# - Review and accept the license agreement (press space to scroll, type 'yes' to accept)
# - Confirm installation path (default is ~/miniconda3, press Enter to confirm)
# - Choose whether to initialize conda (recommend typing 'yes')

# Activate the configuration
source ~/.bashrc

# Remove the installation script (optional)
rm Miniconda3-latest-Linux-x86_64.sh
```

---

## 2. Configure Conda Environment

### Step 2a: Create environment from yml
```bash
# Ensure env_kao310.yml is in your current directory
conda env create -f env_kao310.yml
conda activate kao310
```

### Step 2b: Install rils-rols (requires separate step)
```bash
# rils-rols imports pybind11 in setup.py, but pip's PEP 517 build isolation
# creates a temporary environment that doesn't see conda-installed pybind11.
# --no-build-isolation tells pip to use the current environment's packages.
pip install rils-rols>=1.6.0 --no-build-isolation
```

### Step 2c: Verify installation
```bash
python --version  # Should display Python 3.10.x
python -c "import sklearn, numpy, pandas, shap, deap, gplearn, sympy; print('Core libraries OK')"
python -c "import rils_rols; print('rils-rols OK')"
python -c "from pyoperon.sklearn import SymbolicRegressor; print('pyoperon OK')"
python -c "import kneed; print('kneed OK')"
```

---

## 3. Usage
```bash
# Activate environment before use
conda activate kao310

# Deactivate environment when finished
conda deactivate

# List all conda environments
conda env list

# Remove environment (if needed)
# conda env remove -n kao310
```

---

## Notes

- Installation may take 10-20 minutes depending on network speed
- If conda downloads are slow, consider configuring mirror sources
- The environment includes Julia 1.6+ which is required for PySR functionality
- rils-rols must be installed separately with `--no-build-isolation` due to its
  pybind11 build dependency (see Step 2b)