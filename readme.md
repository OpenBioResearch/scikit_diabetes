# scikit-learn-diabetes

## Overview

This project uses the scikit-learn diabetes dataset which consists of ten baseline variables and 442 diabetes patients. These simple python scripts are designed for exploring disease progression in the context of diabetes.

## Installation and Usage

**Clone the repository:**

```bash
git clone https://github.com/OpenBioResearch/scikit_diabetes.git
cd scikit_diabetes
```

**Create a virtual environment (optional but recommended):**

```bash 
python -m venv .venv
source .venv/bin/activate  # git bash
```

**Activate the virtual environment:**

Windows Git Bash:

```bash
source .venv/Scripts/activate
```
macOS/Linux:

```bash
source .venv/bin/activate
```

**Install the Python dependencies:**

```bash
pip install -r requirements.txt
```

**Run the python scripts:**
```bash
python diabetes_sklearn.py
python diabetes_lasso_features.py
python diabetes_model_comparisons.py
```



## License
This project is licensed under the BSD 3-Clause

