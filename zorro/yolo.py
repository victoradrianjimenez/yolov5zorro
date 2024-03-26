"""
YOLO-specific modules.

Usage:
    $ python -m zorro.yolo --cfg yolov5s.yaml
"""

import sys
import runpy
import zorro.common
sys.modules['models.common'] = sys.modules['zorro.common']

if __name__ == "__main__":
    runpy._run_module_as_main('models.yolo')
