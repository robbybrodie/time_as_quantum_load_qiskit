#!/usr/bin/env python3
"""
connect_ibm.py

Simple script to authenticate against the IBM Quantum platform using the
modern `qiskit-ibm-runtime` package and list available backends.

Usage:
  - Set your token in the environment:
      export IBM_TOKEN="your_token_here"
    or
      export IBMQ_TOKEN="your_token_here"

  - Optionally create a .env file with IBM_TOKEN=... and this script will load it.

  - Run:
      python src/connect_ibm.py
"""

import os
import sys
import traceback

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

def load_token():
    # Load from .env if python-dotenv is available
    if load_dotenv is not None:
        try:
            load_dotenv()  # loads .env automatically if present
        except Exception:
            pass

    token = os.getenv("IBM_TOKEN") or os.getenv("IBMQ_TOKEN")
    return token

def list_backends_with_provider(provider_obj):
    try:
        backends = provider_obj.backends()
    except TypeError:
        # Some provider/service implementations may require no args or different call
        backends = provider_obj.backends
    except Exception as e:
        print("Failed to retrieve backends from provider:", e)
        return

    if not backends:
        print("No backends returned (empty list).")
        return

    print("Available backends:")
    for b in backends:
        # backend name may be callable or attribute
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
            # ignore config errors
            pass

        status = ""
        try:
            st = b.status() if callable(getattr(b, "status", None)) else getattr(b, "status", None)
            if st:
                # status may be an object with 'operational' property
                op = getattr(st, "operational", None)
                if op is not None:
                    status = "operational" if op else "non-operational"
                else:
                    status = str(st)
        except Exception:
            pass

        print(f" - {name}{' (simulator)' if is_sim else ''}{' - ' + status if status else ''}")

def main():
    token = load_token()
    if not token:
        print("IBM Quantum API token not found in environment.")
        print("Set environment variable IBM_TOKEN (or IBMQ_TOKEN) or create a .env file with IBM_TOKEN=...")
        sys.exit(1)

    # Try modern qiskit-ibm-runtime APIs in a couple of ways for compatibility
    tried = []

    # Attempt 1: IBMProvider (qiskit-ibm-runtime)
    try:
        from qiskit_ibm_runtime import IBMProvider  # type: ignore
        print("Using IBMProvider from qiskit_ibm_runtime to authenticate...")
        provider = IBMProvider(token=token)
        list_backends_with_provider(provider)
        return
    except Exception as e:
        tried.append(("IBMProvider", e))
        # continue to try other options

    # Attempt 2: QiskitRuntimeService (alternate modern API)
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore
        print("Using QiskitRuntimeService from qiskit_ibm_runtime to authenticate...")
        service = None
        # Try several common constructor signatures / channel values across versions
        constructor_attempts = [
            {"channel": "ibm_cloud", "token": token},
            {"channel": "ibm_quantum_platform", "token": token},
            {"token": token},
            {}
        ]
        for args in constructor_attempts:
            try:
                service = QiskitRuntimeService(**args) if args else QiskitRuntimeService()
                break
            except TypeError:
                # constructor signature mismatch for this attempt, try next
                continue
            except ValueError as ve:
                # channel or value rejected; try next possible args
                print("QiskitRuntimeService constructor rejected args:", args, "->", ve)
                continue

        if service is None:
            raise RuntimeError("Unable to construct QiskitRuntimeService with known signatures.")
        list_backends_with_provider(service)
        return
    except Exception as e:
        tried.append(("QiskitRuntimeService", e))

    # Attempt 3: legacy qiskit-ibmq-provider (if installed)
    try:
        from qiskit import IBMQ  # type: ignore
        print("Using legacy qiskit.IBMQ to authenticate (fallback)...")
        IBMQ.enable_account(token)
        provider = IBMQ.get_provider(hub=None)  # may raise
        list_backends_with_provider(provider)
        return
    except Exception as e:
        tried.append(("IBMQ (legacy)", e))

    # If we reach here, all attempts failed
    print("Failed to authenticate with available IBM Quantum clients. Attempts:")
    for name, err in tried:
        print(f"- {name}: {type(err).__name__}: {err}")

    print("\nFull traceback of last error:")
    traceback.print_exc()
    sys.exit(2)

if __name__ == "__main__":
    main()
