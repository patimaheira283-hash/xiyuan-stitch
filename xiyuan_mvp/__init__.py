"""Core package for the Xiyuan image-stitching MVP."""

__all__ = ["StitchPipeline"]
__version__ = "0.5.1"


def __getattr__(name):
    if name == "StitchPipeline":
        from .pipeline import StitchPipeline
        return StitchPipeline
    raise AttributeError(name)
