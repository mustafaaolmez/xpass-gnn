"""Label-plugin package for xPass++.

Per ADR-004 (2026-05-13), label production is plugin-based via the
`LabelProducer` protocol in :mod:`src.data.labels.base`. Sprint 1 ships
:class:`src.data.labels.xt_delta.XTDeltaLabels` only. A future VAEP-based
producer is gated by the ADR-004 tripwire.
"""

from src.data.labels.base import LabelProducer  # noqa: F401
