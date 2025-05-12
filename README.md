# DDSP Project Setup

This project uses **DDSP (Differentiable Digital Signal Processing)** with a Conda environment and fast dependency management via **`uv`**.

## Setup Instructions

### Step 1: Create and Activate Conda Environment

First, create the Conda environment with Python 3.9 and activate it.

```bash
conda create -n py39 python=3.9 -y
conda init
```

### Step 2: Restart Terminal

After initializing Conda, close and reopen your terminal.

### Step 3: Activate the Environment

Activate the newly created Conda environment:

```bash
conda activate py39
```

### Step 4: Install `uv` for Fast Dependency Management

Install `uv` for faster package resolution:

```bash
pip install uv
```

### Step 5: Install Project Dependencies

Use `uv` to install all required dependencies from the `requirements.txt` file:

```bash
uv pip install -r requirements.txt
```

---

## Requirements

* **Python 3.9**
* **Conda** (to manage environments)
* **`uv`** (for faster dependency management)