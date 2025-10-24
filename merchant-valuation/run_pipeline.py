"""
Direct entry point for running the pipeline without module syntax.
"""

import sys
from src.cli import main

if __name__ == '__main__':
    sys.exit(main())
