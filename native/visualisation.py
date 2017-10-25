import time
import tkinter
from native.geometry import Point, Size

import numpy as np


class VisualScene:
    """
    Simple visual interface based on TKinter. Please replace with awesome visual interface.
    """
    color_list = ["yellow", "green", "cyan", "magenta"]
    directed_polygon = np.array([[0, -1], [1, 1], [-1, 1]])

    def __init__(self, scene):
        """
        Initializes a visual interface for the simulation. Updates every fixed amount of seconds.
        Represents the scene on a canvas, with the center being (0,0)
        Important: this class progresses the simulation. After each drawing and potential delay,
        the visualisation calls for the progression to the next time step.
        Although it might be cleaner to move that to the simulation manager.
        :param scene: Scene to be drawn. The size of the scene is independent of the size of the visualization
        :return: None
        """
        init_size = Size([1000, 1000])
        self.scene = scene
        self.autoloop = True
        self.delay = 10
        self.window = tkinter.Tk()
        self.window.title("SpaceDust")
        self.window.geometry("%dx%d" % (init_size[0], init_size[1]))
        self.canvas = tkinter.Canvas(self.window, width=init_size[0], height=init_size[1])
        self.canvas.pack(fill=tkinter.BOTH, expand=1)
        self.canvas.delete('all')

    @property
    def size(self):
        return Size([self.canvas.winfo_width(), self.canvas.winfo_height()])

    @size.setter
    def size(self, value):
        self.window.geometry("%dx%d" % tuple(value))

    def start(self):
        self.loop()
        self.window.mainloop()

    def disable_loop(self):
        """
        Stop automatically redrawing.
        Enables space and Left-mouse click as progressing simulation.
        :return: None
        """
        self.autoloop = False
        self.window.bind("<Button-1>", self.loop)
        self.window.bind("<space>", self.loop)

    def loop(self, _=None):
        """
        Public interface for visual scene loop. If required, has a callback reference to itself to keep the simulation going.
        :param _: Event object from tkinter
        :return: None
        """
        if self.scene.status == 'DONE':
            self.window.destroy()
            self.autoloop = False
        else:
            self.scene.step()
            self.draw_scene()
            if self.autoloop:
                self.window.after(self.delay, self.loop)
            else:
                self.loop()

    def _give_relative_position(self, event):
        """
        Give the position of an event (most probably a mouse click)
        :param event: some mouse click
        :return: Position in Carthesian coordinates in [0,1]
        """
        x, y = (event.x / self.size[0], 1 - event.y / self.size[1])
        print("Mouse location: (%.2f,%.2f)" % (x, y))

    def draw_scene(self):
        """
        Method that orders the draw commands of all objects within the scene.
        All objects are removed prior to the drawing step.
        :return: None
        """
        self.canvas.delete('all')
        self.draw_particles()

    def store_scene(self, _, filename=None):

        directory = 'images'
        if not filename:
            import time

            name = "scene#%d" % time.time()
            filename = "%s/%s-%.2f.eps" % (directory, name, self.scene.time)
        print("Snapshot at %.2f. Storing in %s" % (self.scene.time, filename))
        self.canvas.postscript(file=filename, pageheight=self.size[1], pagewidth=self.size[0])

    def draw_particles(self):
        """
        Draws all the particles in the scene using the visual_particles coordinates.
        :return: None
        """
        start_pos_array, end_pos_array = self.get_visual_particle_coordinates()
        for index in range(self.scene.num_particles):
            self.canvas.create_oval(start_pos_array[index, 0], start_pos_array[index, 1],
                                    end_pos_array[index, 0], end_pos_array[index, 1],
                                    fill='black')

    def get_visual_particle_coordinates(self):
        """
        Computes the coordinates of all particle relative to the visualization.
        Uses vectorized operations for speed increments
        :return: relative start coordinates, relative end coordinates.
        """
        rel_pos_array = (self.scene.position_array) / self.scene.size.array + np.array([0.5,0.5])
        rel_size_array = np.ones(
            self.scene.position_array.shape) * self.scene.particle_size / self.scene.size.array * self.size.array
        vis_pos_array = np.hstack((rel_pos_array[:, 0][:, None], 1 - rel_pos_array[:, 1][:, None])) * self.size.array
        start_pos_array = vis_pos_array - 0.5 * rel_size_array
        end_pos_array = vis_pos_array + 0.5 * rel_size_array
        return start_pos_array, end_pos_array

    def convert_relative_coordinate(self, coord):
        """
        Converts relative coordinates (from [0,1]x[0,1]) to screen size coordinates.
        Should raise an error when coordinates fall from scene,
        but this method is used so frequently I'd rather not make the computation
        Also changes the orientation to a Carthesian coordinate system
        :param coord: coordinates (fractions to be converted)
        :return: a Size with the coordinates of screen
        """
        return Size([0.5 + coord[0], 1.5 - coord[1]]) * self.size
