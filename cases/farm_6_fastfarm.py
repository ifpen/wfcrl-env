"""
2 x 3 Layout. 
Used on FAST.Farm
"""

Cx = [0.0, 504.0, 1008.0, 0.0, 504.0, 1008.0]
Cy = [-252, -252, -252, 252, 252, 252]

dt = 3
buffer_window = 600

interface_kwargs = {"measurement_window": int(buffer_window/dt)}