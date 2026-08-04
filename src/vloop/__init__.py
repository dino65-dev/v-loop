"""Verified Adaptive Loop Agent.

The root exports the minimal application API.  Domain-specific integrations
should use ``vloop.control``, ``vloop.evidence``, ``vloop.execution``,
``vloop.intelligence``, or ``vloop.governance`` for an explicit dependency
boundary.
"""

from .controller import VerifiedLoop
from .ledger import EvidenceLedger
from .models import ActionIntent, ActionRule, TaskContract
from .policy import PolicyGate

__all__ = [
    "ActionIntent",
    "ActionRule",
    "EvidenceLedger",
    "PolicyGate",
    "TaskContract",
    "VerifiedLoop",
]
