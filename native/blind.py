from native.visualisation import VisualScene
import time


class NoVisualScene(VisualScene):
    def __init__(self, scene,max_iterations=0):
        self.scene = scene
        self.start_time = 0
        self.end_time = 0

    def start(self):
        self.start_time = time.time()
        while not self.scene.status == 'DONE':
            try:
                self.scene.step()
                # print("Iteration took %.4f seconds" % (time.time() - self.time))
                # print("Time step %d" % self.scene.counter)
            except KeyboardInterrupt:
                self.end_time = time.time()
                print("\nUser interrupted simulation")
                time_spent = self.end_time - self.start_time
                print("%d iterations in %.2f seconds (%.4e per iteration)" % (
                self.scene.counter, time_spent, time_spent / self.scene.counter))
                self.scene.status = 'DONE'
