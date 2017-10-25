import numpy as np

from native.geometry import Point, Size


class Scene:
    """
    Models a scene. A scene is a rectangular object particles inside.
    """

    def __init__(self, num_particles):
        """
        Initializes a Scene the initial distributions of particles
        :return: scene instance.
        """
        self.time = 0
        self.dt = 0.05
        self.counter = 0
        self.size = Size([100,100])
        self.num_particles = num_particles

        # Array initialization
        self.position_array = np.zeros([self.num_particles, 2])
        self.last_position_array = np.zeros([self.num_particles, 2])
        self.velocity_array = np.zeros([self.num_particles, 2])
        self.force_array = np.zeros([self.num_particles, 2])
        self.particle_size = np.zeros([self.num_particles,2])
        self.active_entries = np.ones(self.num_particles, dtype=bool)

        # Parameters
        d = 3
        v = 1
        self._init_particles(d,v)
        self.status = 'RUNNING'

    def _init_particles(self, d, v):
        """
        Protected method that determines how the particles are initially distributed,
        as well as with what properties they come. Overridable.
        :param: Initial number of particles
        :return: None
        """
        # Positions
        r = np.random.normal(loc=d, scale=d / 4, size=self.num_particles)
        thetas = np.random.rand(self.num_particles) * 2 * np.pi
        self.position_array[:, 0] = r * np.cos(thetas)
        self.position_array[:, 1] = r * np.sin(thetas)

        # Velocities
        rn = np.zeros((self.num_particles, 3))
        zs = np.zeros(rn.shape)
        rn[:,:2] = self.position_array
        zs[:, 2] = v
        self.velocity_array = np.cross(zs,rn)[:,:2]

        self.force_array[:] = np.random.random(self.position_array.shape)
        self.particle_size[:] = np.random.random(self.position_array.shape)+1

    def is_within_boundaries(self, coord: Point):
        """
        Check whether a single point lies within the scene.
        :param coord: Point under consideration
        :return: True if within scene, false otherwise
        """
        within_boundaries = all(np.array([0, 0]) < coord.array) and all(coord.array < self.size.array)
        return within_boundaries

    def move_particles(self):
        """
        Performs a vectorized move of all the particles.
        Assumes that all the velocities have been set accordingly.
        :return: None
        """
        self.time += self.dt
        self.counter += 1
        self.velocity_array += self.force_array * self.dt
        self.last_position_array = np.array(self.position_array)
        self.position_array += self.velocity_array * self.dt
