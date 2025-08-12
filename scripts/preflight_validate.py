#!/usr/bin/env python3
"""
Preflight validation runner (Python).

This script performs a repeatable set of checks before you submit jobs to real IBM Quantum hardware:
  - Loads environment variables from a .env file (if present).
  - Verifies required Python packages are importable; prints the pip install command if something's missing.
  - Optionally attempts to authenticate to IBM Quantum and lists available backends (network).
  - Runs a small simulator-based validation of a minimal quantum circuit using AerSimulator.

Usage:
  # Basic checks + simulator test (no network):
  python scripts/preflight_validate.py --no-network

  # Do everything (requires IBM_TOKEN in env or .env):
  python scripts/preflight_validate.py

  # Target a specific IBM Runtime instance (if you use instances):
  python scripts/preflight_validate.py --instance <instance-name>

  # Show verbose output:
  python scripts/preflight_validate.py --verbose

Exit codes:
  0  => all checks passed
  1  => missing required packages
  2  => missing IBM token (when network checks requested)
  3  => authentication / backend listing failed
  4  => simulator validation failed
"""

from __future__ import annotations
import os
import sys
import argparse
import importlib
import traceback
from pathlib import Path
from typing import List, Tuple

# Try to load dotenv if available; this is optional but helpful
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Project root (two levels up from this file when run from scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_IMPORTS = [
    ("qiskit_ibm_runtime", "QiskitRuntimeService"),  # modern runtime client
    ("qiskit", None),                                # core Qiskit
    # Aer may be available as the namespaced package `qiskit.providers.aer`
    # or as the standalone `qiskit_aer` package depending on installation.
    # Represent Aer as a group of alternatives; the check_imports function
    # will consider the group satisfied if any alternative imports successfully.
    # Use module-only checks for alternatives (symbol lookup may vary by package layout).
    [("qiskit.providers.aer", None), ("qiskit_aer", None)],  # simulator (either)
    ("numpy", None),
]

PIP_INSTALL_CMD = f"pip install -r {REPO_ROOT / 'requirements.txt'}"

def load_env():
    if load_dotenv is not None:
        try:
            load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
            print("Loaded .env (if present).")
        except Exception:
            print("Warning: python-dotenv present but failed to load .env (continuing).")
    else:
        # If python-dotenv isn't installed, we still allow the OS environment to provide variables
        print("python-dotenv not installed — relying on environment variables only.")


def check_imports() -> Tuple[bool, List[Tuple[str, str]]]:
    """
    Attempt to import required modules. Returns (all_ok, missing_list).

    REQUIRED_IMPORTS may contain:
      - tuple(mod_name, symbol) for a single required import, or
      - a list of tuples [(mod, symbol), (alt_mod, alt_symbol), ...] representing
        alternative imports (satisfied if any alternative imports successfully).

    missing_list is a list of tuples (module_or_alternatives, reason).
    """
    missing: List[Tuple[str, str]] = []
    for entry in REQUIRED_IMPORTS:
        # If entry is a list, treat it as alternatives
        if isinstance(entry, list):
            alternatives = entry
            ok = False
            alt_errors: List[Tuple[str, str]] = []
            for mod, symbol in alternatives:
                try:
                    m = importlib.import_module(mod)
                    if symbol and not hasattr(m, symbol):
                        raise ImportError(f"module present but missing symbol {symbol}")
                    ok = True
                    break
                except Exception as e:
                    alt_errors.append((mod, str(e)))
                    continue
            if not ok:
                alt_names = " | ".join([a for a, _ in alternatives])
                # produce a concise summary of all alternative import errors
                reason = "; ".join([f"{m}: {err}" for m, err in alt_errors]) if alt_errors else "failed to import any alternatives"
                missing.append((alt_names, reason))
        else:
            mod, symbol = entry
            try:
                m = importlib.import_module(mod)
                if symbol and not hasattr(m, symbol):
                    missing.append((mod, f"module present but missing symbol {symbol}"))
            except Exception as e:
                missing.append((mod, str(e)))
    all_ok = len(missing) == 0
    return all_ok, missing


def try_authenticate_and_list_backends(token: str | None, instance: str | None, channel: str | None, verbose: bool = False) -> bool:
    """
    Attempts to authenticate using modern Qiskit runtime APIs and list backends.
    Returns True on success, False otherwise.

    New parameter:
      - channel: optional channel name (e.g. 'ibm_cloud' or 'ibm_quantum_platform') to try when constructing QiskitRuntimeService.
    """
    tried = []

    # Attempt 1: QiskitRuntimeService (modern)
    if token is None:
        print("No IBM token provided; skipping authentication attempts.")
        return False

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore
        print("Trying QiskitRuntimeService from qiskit_ibm_runtime...")
        service = None

        # Try a number of constructor signatures / argument combinations commonly
        # used across qiskit-ibm-runtime versions.
        constructor_attempts = []
        if channel:
            constructor_attempts.append({"channel": channel, "token": token})
        if instance:
            constructor_attempts.append({"instance": instance, "token": token})
        # token-only attempt
        constructor_attempts.append({"token": token})
        # no-arg attempt
        constructor_attempts.append({})
        for args in constructor_attempts:
            try:
                if args:
                    service = QiskitRuntimeService(**args)
                else:
                    service = QiskitRuntimeService()
                break
            except TypeError:
                continue
            except Exception as exc:
                if verbose:
                    print("QiskitRuntimeService attempt rejected args:", args, "->", type(exc).__name__, exc)
                continue

        if service is None:
            raise RuntimeError("Unable to construct QiskitRuntimeService with known signatures.")

        # list backends
        try:
            backends = service.backends()
        except TypeError:
            backends = service.backends
        if not backends:
            print("No backends returned (empty list).")
        else:
            print("Available backends:")
            for b in backends:
                try:
                    name = b.name() if callable(getattr(b, "name", None)) else getattr(b, "name", str(b))
                except Exception:
                    name = str(b)
                is_sim = False
                try:
                    cfg = b.configuration() if callable(getattr(b, "configuration", None)) else getattr(b, "configuration", None)
                    if cfg is not None:
                        is_sim = getattr(cfg, "simulator", False)
                except Exception:
                    pass
                status = ""
                try:
                    st = b.status() if callable(getattr(b, "status", None)) else getattr(b, "status", None)
                    if st:
                        op = getattr(st, "operational", None)
                        if op is not None:
                            status = "operational" if op else "non-operational"
                        else:
                            status = str(st)
                except Exception:
                    pass
                print(f" - {name}{' (simulator)' if is_sim else ''}{' - ' + status if status else ''}")
        return True
    except Exception as e:
        tried.append(("QiskitRuntimeService", e))

    # Attempt 2: IBMProvider (alternate modern API)
    try:
        from qiskit_ibm_runtime import IBMProvider  # type: ignore
        print("Trying IBMProvider from qiskit_ibm_runtime...")
        provider = IBMProvider(token=token)
        try:
            backends = provider.backends()
        except TypeError:
            backends = provider.backends
        print("Available backends (IBMProvider):")
        for b in backends:
            try:
                name = b.name() if callable(getattr(b, "name", None)) else getattr(b, "name", str(b))
            except Exception:
                name = str(b)
            print(f" - {name}")
        return True
    except Exception as e:
        tried.append(("IBMProvider", e))

    # Attempt 3: legacy qiskit-IBMQ
    try:
        from qiskit import IBMQ  # type: ignore
        print("Trying legacy qiskit.IBMQ...")
        IBMQ.enable_account(token)
        provider = IBMQ.get_provider(hub=None)
        try:
            backends = provider.backends()
        except TypeError:
            backends = provider.backends
        print("Available backends (IBMQ legacy):")
        for b in backends:
            try:
                name = b.name() if callable(getattr(b, "name", None)) else getattr(b, "name", str(b))
            except Exception:
                name = str(b)
            print(f" - {name}")
        return True
    except Exception as e:
        tried.append(("IBMQ (legacy)", e))

    print("Failed to authenticate with available IBM Quantum clients. Attempts:")
    for name, err in tried:
        print(f"- {name}: {type(err).__name__}: {err}")
    if tried:
        print("\nLast traceback:")
        traceback.print_exception(type(tried[-1][1]), tried[-1][1], tried[-1][1].__traceback__)
    return False


def run_simulator_validation(verbose: bool = False) -> bool:
    """
    Run a tiny circuit on AerSimulator to validate the local simulator stack.
    Returns True if successful.
    """
    try:
        # Import lazily so we only require qiskit when this is executed.
        # AerSimulator may be available via `qiskit.providers.aer` or the
        # standalone `qiskit_aer` package depending on the environment.
        from qiskit import QuantumCircuit, transpile
        try:
            from qiskit.providers.aer import AerSimulator  # preferred
        except Exception:
            # fallback to standalone qiskit_aer
            try:
                from qiskit_aer import AerSimulator  # type: ignore
            except Exception as e:
                raise e
    except Exception as e:
        print("Simulator imports failed:", e)
        return False

    try:
        print("Building minimal validation circuit (1 qubit, X then measure)...")
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)

        sim = AerSimulator()
        t_qc = transpile(qc, sim)
        job = sim.run(t_qc)
        result = job.result()
        counts = result.get_counts()
        print("Simulator counts:", counts)
        # Expect mostly '1' (or exactly {'1': 1} for a single shot)
        if not counts:
            print("Simulator returned no counts.")
            return False
        print("Simulator validation succeeded.")
        return True
    except Exception as e:
        print("Simulator validation failed:", type(e).__name__, e)
        if verbose:
            traceback.print_exc()
        return False


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Preflight validate environment, IBM auth, and simulator")
    parser.add_argument("--no-network", action="store_true", help="Skip IBM authentication / network checks")
    parser.add_argument("--instance", type=str, default=None, help="IBM Runtime instance name to target (optional)")
    parser.add_argument("--channel", type=str, default=None, help="IBM Runtime channel name to target (optional, e.g. 'ibm_cloud')")
    parser.add_argument("--verbose", action="store_true", help="Verbose output for debugging")
    args = parser.parse_args(argv)

    print(f"Project root: {REPO_ROOT}")
    load_env()

    print("\n1) Checking required Python packages...")
    all_ok, missing = check_imports()
    if not all_ok:
        print("Missing or broken packages detected:")
        for mod, reason in missing:
            print(f" - {mod}: {reason}")

        # Try to produce one-line installation options (from requirements.txt if present)
        req_path = REPO_ROOT / "requirements.txt"
        per_pkg_cmds: List[str] = []
        if req_path.exists():
            try:
                raw = req_path.read_text().splitlines()
                for line in raw:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    # keep the exact requirement spec for a precise install line
                    per_pkg_cmds.append(f'pip install "{s}"  # installs {s.split("=")[0].split(">")[0].split("<")[0].strip()}')
            except Exception:
                per_pkg_cmds = []

        print("\nInstallation options (one line per option):")
        # Option 1: install all requirements at once
        print(f" - Install all required packages from requirements.txt:")
        print(f"     {PIP_INSTALL_CMD}")
        # Option 2: install individual packages with pinned/flexible spec from requirements.txt (if available)
        if per_pkg_cmds:
            print(" - Install individual packages (one line per package):")
            for cmd in per_pkg_cmds:
                print(f"     {cmd}")
        else:
            # Fallback: suggest installing each missing module by name
            print(" - Install individual packages (fallback):")
            for mod, _ in missing:
                print(f"     pip install {mod}")

        print("\nAfter installing, re-run:")
        print("  python3 scripts/preflight_validate.py")
        sys.exit(1)
    else:
        print("All required packages appear importable.")

    # Load token (IBM_TOKEN or IBMQ_TOKEN)
    token = os.getenv("IBM_TOKEN") or os.getenv("IBMQ_TOKEN")
    if args.no_network:
        print("\n--no-network specified: skipping IBM Quantum authentication checks.")
    else:
        print("\n2) Attempting IBM Quantum authentication and backend listing...")
        if not token:
            print("IBM token not found in environment.")
            print("Missing items and one-line fixes (each on its own line):")
            print(" - Set token for current shell:")
            print('     export IBM_TOKEN="<your_token_here>"')
            print(" - Or write token into .env in project root:")
            print("     echo 'IBM_TOKEN=<your_token_here>' > .env")
            print(" - Or set IBMQ_TOKEN if preferred:")
            print('     export IBMQ_TOKEN="<your_token_here>"')
            print("\nAfter setting the token, re-run:")
            print("  set -a && source .env && set +a && python3 scripts/preflight_validate.py")
            sys.exit(2)

        ok = try_authenticate_and_list_backends(token=token, instance=args.instance, channel=args.channel, verbose=args.verbose)
        if not ok:
            print("Authentication/backend listing failed. Diagnosing common causes and one-line fixes:")
            print(" - Token invalid or expired: re-set token (one-liners):")
            print('     export IBM_TOKEN="<your_new_token_here>"')
            print("     echo 'IBM_TOKEN=<your_new_token_here>' > .env")
            print(" - No default runtime instance set (if using instances). Options:")
            print("     # Option A: specify instance for this run")
            print("     set -a && source .env && set +a && python3 scripts/preflight_validate.py --instance <instance-name>")
            print("     # Option B: set default instance via IBM Cloud / SDK configuration (see IBM docs).")
            print(" - Missing packages (if authentication client not installed):")
            print(f"     {PIP_INSTALL_CMD}")
            print(" - Network/firewall blocking outbound connections to IBM Quantum endpoints.")
            print("\nFor verbose debug, re-run with --verbose to see exception traces.")
            sys.exit(3)

    print("\n3) Running local simulator validation (AerSimulator)...")
    sim_ok = run_simulator_validation(verbose=args.verbose)
    if not sim_ok:
        print("Simulator validation failed.")
        sys.exit(4)

    print("\nPreflight validation completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
