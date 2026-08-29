"""Offline experience compiler for CompilerKV decision tables."""

from .cql import CQLConfig, fit_horizon_one_cql
from .records import CalibrationRecord, load_records


def compile_calibration_records(*args, **kwargs):
    """Lazily import the CLI module so ``python -m`` remains warning-free."""

    from .compile import compile_calibration_records as compile_records

    return compile_records(*args, **kwargs)

__all__ = [
    "CQLConfig",
    "CalibrationRecord",
    "compile_calibration_records",
    "fit_horizon_one_cql",
    "load_records",
]
