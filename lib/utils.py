"""@package NMF-visualization

Utility functions.
"""

import os

from pathlib import Path

def createDir(path):
    """
    Make sure a path's parent directories exist.
    
    Keyword arguments:
    path -- the path to check.
    """
    
    dirPath = os.path.dirname(path)
    Path(dirPath).mkdir(parents = True, exist_ok = True)
