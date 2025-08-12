# time_as_quantum_load_qiskit — experimental

This repository is an experimental playground to explore unifying SR and GR by modelling time-dilation as quantum/computational load. The first pass sets up a Python virtual environment and a minimal connection from Qiskit to IBM Quantum (the new IBM Quantum platform).

This README documents a safe, repeatable setup that uses the current IBM Quantum Python packages.

WARNING: keep your IBM Quantum API token secret. Do not commit it to git. Use environment variables or a local .env file that's listed in .gitignore.

## What this repo provides (initial)
- A short Python script (under `src/`) that tries to connect to IBM Quantum using the modern IBM packages.
- A `requirements.txt` that pins the packages we recommend to install (qiskit and ibm runtime package).
- Guidance for creating and using a venv and how to provide the API token.

## Prerequisites
- macOS (instructions use zsh)
- Python 3.8+ (use `python3 --version` to check)
- Network access to IBM Quantum services

## Setup (macOS / zsh)
1. Clone the repo (already done if you're here):
   - git clone https://github.com/robbybrodie/time_as_quantum_load_qiskit.git

2. Create and activate a virtual environment:
   - python3 -m venv venv
   - source venv/bin/activate

3. Install dependencies:
   - pip install --upgrade pip
   - pip install -r requirements.txt

4. Provide your IBM Quantum token securely:
   Recommended: export the token in your shell session (temporary):
   - export IBM_TOKEN="YOUR_IBM_QUANTUM_TOKEN_HERE"
     (or set it to your token)
   Alternatively create a local `.env` file and load it (do not commit `.env`).

5. Run the connection test:
   - python src/connect_ibm.py
   The script will attempt to authenticate with the installed IBM package(s) and list available backends (or explain what to fix).

## Files to be created
- `requirements.txt` — lists python packages to install (qiskit + IBM runtime)
- `src/connect_ibm.py` — a small script to authenticate and list available backends
- `.gitignore` — includes `venv/` and any token files
- `.env.example` — example for how to store your token locally (not tracked)

## Security notes
- Never commit your token to this repo or any public place.
- If you accidentally commit a token, rotate it immediately from IBM Quantum account settings.

## Next steps
After the basic connection is working, we will:
- Add examples that run small circuits on simulator and hardware via IBM runtime.
- Add tests and scripts to experiment with computational load vs. simulated time-dilation metrics.

## IBM Quantum runtime & token

How to get your IBM Quantum API token
1. Sign in to the IBM Quantum dashboard at https://quantum-computing.ibm.com.
2. Open your account settings (click your avatar / account menu) and look for "API token" or "Credentials".
3. If you don't have a token yet, create one from that page. Treat this token like a password.

Set the token as an environment variable (recommended)
- In zsh (temporary for the session):
  - export IBM_QUANTUM_TOKEN="YOUR_REAL_IBM_QUANTUM_TOKEN"
- To persist between sessions, add the export line to your shell profile (e.g., `~/.zshrc`) or use a local `.env` loaded by your environment. Never commit the token or any file containing it.

What we use in this project
- This project uses the `qiskit-ibm-runtime` package and the IBM runtime programming model (Session, Sampler, Estimator). See the qiskit-ibm-runtime project and docs for details:
  - qiskit-ibm-runtime GitHub: https://github.com/Qiskit/qiskit-ibm-runtime
  - Qiskit runtime docs: https://qiskit.org/documentation/partners/qiskit_ibm_runtime/
  - IBM Quantum docs (overview & account): https://quantum-computing.ibm.com/docs

Configuration files added
- `configs/runtime.example.yaml` — example runtime configuration (contains `ibmq.instance`, `ibmq.token_env`, runtime defaults, and backend placeholder).
- `configs/local.overrides.example.yaml` — per-user override template. Copy to `configs/local.overrides.yaml` for local, private overrides. This file is listed in `.gitignore` by default.

Security reminder
- The actual IBM Quantum API token must never be committed to GitHub or shared. If a token is accidentally exposed, rotate it immediately from the IBM Quantum dashboard.
