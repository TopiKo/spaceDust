#!/usr/bin/env python3
from native.visualisation import VisualScene
from native.blind import NoVisualScene
from native.scene import Scene

scene = Scene(600, False)
#vis = NoVisualScene(scene)
vis = VisualScene(scene)

vis.start()
