import numpy as np

from native.geometry import Point, Size


class Scene:
    """
    Models a scene. A scene is a rectangular object with particles inside.
    """

    def __init__(self, num_particles):
        """
        Initializes a Scene the initial distributions of particles
        :return: scene instance.
        """
        self.time = 0
        self.dt = 0.005
        self.counter = 0
        self.size = Size([100, 100])
        self.num_particles = num_particles

        # Array initialization
        self.position_array = np.zeros([self.num_particles, 2])
        self.velocity_array = np.zeros([self.num_particles, 2])
        self.accel_array = np.zeros([self.num_particles, 2])
        self.force_array = np.zeros([self.num_particles, 2])
        self.particle_size = np.zeros([self.num_particles, 2])
        self.masses = np.ones(self.num_particles)

        self.pos_difference = np.zeros((self.num_particles, self.num_particles, 2))
        self.mask = 4
        # Parameters
        self.g = 30
        self._init_particles()
        self.status = 'RUNNING'

    def _init_particles(self):
        """
        Protected method that determines how the particles are initially distributed,
        as well as with what properties they come. Overridable.
        :param: Initial number of particles
        :return: None
        """
        # Positions
        self.position_array[:] = self.size.array*(0.5 - np.random.random([self.num_particles,2]))
        # Velocities
        rn = np.zeros((self.num_particles, 3))
        zs = np.zeros(rn.shape)
        rn[:, :2] = self.position_array
        zs[:, 2] = 1. / 100
        self.velocity_array = np.cross(zs, rn)[:, :2]

        self.particle_size[:] = np.random.random(self.position_array.shape) + 1

        # Masses
        self.tiled_masses = np.tile(self.masses, self.num_particles).reshape((self.num_particles, self.num_particles))
        np.fill_diagonal(self.tiled_masses, 0)
        self.tiled_masses = self.tiled_masses[:, :, None]  # Probably not a time saver

    def is_within_boundaries(self, coord: Point):
        """
        Check whether a single point lies within the scene.
        :param coord: Point under consideration
        :return: True if within scene, false otherwise
        """
        within_boundaries = all(np.array([0, 0]) < coord.array) and all(coord.array < self.size.array)
        return within_boundaries

    def update_directions(self):
        trans_pos = np.tile(self.position_array, self.num_particles).reshape(
            (self.num_particles, self.num_particles, 2))
        pos = np.transpose(trans_pos, (1, 0, 2))
        self.pos_difference[:] = pos - trans_pos
        self.pos_difference[np.arange(self.num_particles), np.arange(self.num_particles), :] = self.mask

    def update_forces(self):
        distances = np.linalg.norm(self.pos_difference, axis=2, keepdims=True)  # *masses
        planet_contributions = self.pos_difference * self.tiled_masses / distances ** 3
        self.accel_array = planet_contributions.sum(axis=1) * self.g

    def update_pos_and_velo(self):
        """
        Performs a vectorized move of all the particles.
        Assumes that all the velocities have been set accordingly.
        :return: None
        """
        self.time += self.dt
        self.counter += 1
        self.velocity_array += self.accel_array * self.dt
        self.position_array += self.velocity_array * self.dt
        self.position_array = np.ma.array(self.position_array, mask=False)

    def step(self):
        self.time += self.dt
        self.counter += 1
        self.update_directions()
        self.update_forces()
        self.update_pos_and_velo()
