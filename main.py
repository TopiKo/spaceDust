#!/usr/bin/env python3
from native.visualisation import VisualScene
from native.scene import Scene

scene = Scene(100)
vis = VisualScene(scene)

vis.start()
