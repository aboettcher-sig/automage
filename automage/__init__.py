"""AutoMage image classification with SAM3 features and optional superpixels."""
from .config import Config,load_dictionary

def classify(*args,**kwargs):
    from .pipeline import classify as run
    return run(*args,**kwargs)
