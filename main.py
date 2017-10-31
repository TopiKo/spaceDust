#!/usr/bin/env python3
from native.visualisation import VisualScene
from native.blind import NoVisualScene
from native.scene import Scene

scene = Scene(4, True)
#vis = NoVisualScene(scene)
vis = VisualScene(scene)

vis.start()
