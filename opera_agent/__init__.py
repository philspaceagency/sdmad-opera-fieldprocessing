"""OpERA field-processing agent: natural-language requests → geotagged GoPro frames."""
from . import pipeline
from .pipeline import inspect, process_survey

__all__ = ["pipeline", "inspect", "process_survey", "OperaAgent"]


def __getattr__(name):
    if name == "OperaAgent":          # lazy: the pipeline works without the google-genai package
        from .agent import OperaAgent
        return OperaAgent
    raise AttributeError(name)
