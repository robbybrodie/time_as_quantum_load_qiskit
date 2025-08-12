"""
Capacity-Time Dilation Quantum Experiments

This package implements quantum emulations to test the hypothesis that
computational demand creates time dilation effects in quantum systems.

The core idea: N = C/D where C is capacity, D is demand, and N controls
local clock rate via dτ = N dt.
"""

__version__ = "0.1.0"

from . import clocks
from . import qfi 
from . import circuits
from . import demand_capacity
from . import backreaction
from . import response

__all__ = [
    "clocks",
    "qfi", 
    "circuits",
    "demand_capacity",
    "backreaction", 
    "response"
]
