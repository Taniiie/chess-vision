"""
Lists the actual NAMES of your camera devices (not just numbers),
so we can identify exactly which index corresponds to DroidCam.

Requires: pip install pygrabber
"""

from pygrabber.dshow_graph import FilterGraph

graph = FilterGraph()
devices = graph.get_input_devices()

print("Camera devices found (index: name):")
for i, name in enumerate(devices):
    print(f"  {i}: {name}")
