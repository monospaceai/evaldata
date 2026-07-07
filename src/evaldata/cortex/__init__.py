"""Snowflake Cortex Analyst evaluation.

`CortexAnalystSolver` is a `Solver` that sends a question to the Cortex Analyst REST endpoint and
returns the SQL it generates; evaldata executes that SQL on the Snowflake adapter and scores it
through the usual cascade. `CortexAnalystClient` is the HTTP seam it sends through.
"""

from evaldata.cortex.client import CortexAnalystClient, CortexTransport
from evaldata.cortex.solver import CortexAnalystSolver

__all__ = ["CortexAnalystClient", "CortexAnalystSolver", "CortexTransport"]
