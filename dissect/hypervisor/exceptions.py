from __future__ import annotations


class Error(Exception):
    pass


class InvalidHeaderError(Error):
    pass


class InvalidSignature(Error):
    pass


class InvalidVirtualDisk(Error):
    pass


class ASIFRecoveryError(Error):
    """Raised when a requested ASIF recovery operation cannot be satisfied.

    This includes selecting a directory version that cannot be read, or opening data when no
    readable directory is available. It is never raised for a metadata-only failure.
    """
