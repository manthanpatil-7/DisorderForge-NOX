"""DisorderForge-NOX — CAID 4 disorder predictor (CPU, precomputed embeddings)."""
__all__ = ["DisorderForgeRM"]
__version__ = "1.0.0"


def __getattr__(name):
    # Lazy: importing the package (e.g. for the pure _core/io modules) must not
    # pull in torch. DisorderForgeRM is loaded only when actually requested.
    if name == "DisorderForgeRM":
        from .model import DisorderForgeRM
        return DisorderForgeRM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
