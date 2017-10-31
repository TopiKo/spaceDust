import numpy as np

from native.geometry import Point, Size


class Scene:
    """
    Models a scene. A scene is a rectangular object with particles inside.
    """

    def __init__(self, num_particles, test = False):
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
        self.masses = np.ones(self.num_particles) + np.random.rand(self.num_particles)*.1

        self.pos_difference = np.zeros((self.num_particles, self.num_particles, 2))
        self.mask = 4
        self.exclude_mask = np.zeros(self.num_particles).astype(bool)
        # Parameters
        self.g = 30
        self.density = .5
        self._init_particles(test)
        self.status = 'RUNNING'

        # Much used arrays
        self.range = np.arange(self.num_particles)
        self.ones = np.ones(self.num_particles)[:, None]

    def _init_particles(self, test):
        """
        Protected method that determines how the particles are initially distributed,
        as well as with what properties they come. Overridable.
        :param: Initial number of particles
        :return: None
        """
        # Positions
        self.position_array[:] = self.size.array * (0.5 - np.random.random([self.num_particles, 2]))
        if test:
            self.position_array[:] = self.size.array * [[0.03,0.03], [-.03,-.03],[0.06,-0.06], [-.06,.06]]


        # Velocities
        rn = np.zeros((self.num_particles, 3))
        zs = np.zeros(rn.shape)
        rn[:, :2] = self.position_array
        zs[:, 2] = 1. / 100

        if test: zs[:, 2] = 0

        self.velocity_array = np.cross(zs, rn)[:, :2]

        self.particle_size[:] = np.sqrt(self.masses[:,np.newaxis])*self.density #np.random.random(self.position_array.shape) + 1

        # Masses
        self.tiled_masses = np.tile(self.masses, self.num_particles).reshape((self.num_particles, self.num_particles))
        np.fill_diagonal(self.tiled_masses, 0)

        print(self.tiled_masses)

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
        wild_pos = np.transpose(trans_pos, (1, 0, 2))
        self.pos_difference = wild_pos - trans_pos
        self.pos_difference[self.range, self.range, :] = self.mask

    def update_forces(self):
        squared_distances = self.pos_difference[:, :, 0] ** 2 + self.pos_difference[:, :, 1] ** 2
        planet_contributions = self.pos_difference * (self.tiled_masses / (squared_distances * np.sqrt(
            squared_distances)))[:, :, None]
        self.accel_array = np.hstack((np.dot(planet_contributions[:, :, 0], self.ones),
                                      np.dot(planet_contributions[:, :, 1], self.ones))) * self.g

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

    def update_masses(self):
        """
        See whether two particles are within threshold. If so puts the mass of lighter one to heavier one
        and performs collsion where momentum is conserved according to p1_i + p2_i = p_end =>
        v_end = (m1*v1_i + m2*v2_i)/(m1 + m2)
        """

        sizesnn = np.tile(self.particle_size[:,0], self.num_particles).reshape(
            (self.num_particles, self.num_particles))
        sizesnnT = np.transpose(sizesnn)
        threshold = sizesnn + sizesnnT


        collision_idxs = np.where((self.pos_difference[:, :, 0] ** 2 + \
            self.pos_difference[:, :, 1] ** 2) < threshold**2)


        first_smaller = self.masses[collision_idxs[0]] <= self.masses[collision_idxs[1]]
        annihilate = collision_idxs[0][first_smaller]
        survive = collision_idxs[1][first_smaller]

        if len(survive) != 0 or len(annihilate) != 0:
            self.position_array[survive] = (self.position_array[survive]*self.masses[survive, np.newaxis] + \
                                            self.position_array[annihilate]*self.masses[annihilate, np.newaxis])/ \
                                            (self.masses[survive, np.newaxis] + self.masses[annihilate, np.newaxis])

            self.velocity_array[survive] = (self.masses[annihilate, np.newaxis]*self.velocity_array[annihilate] + \
                                            self.masses[survive, np.newaxis]*self.velocity_array[survive])/ \
                                            (self.masses[survive, np.newaxis] + self.masses[annihilate, np.newaxis])

            self.masses[survive] += self.masses[annihilate]
            self.exclude_mask[annihilate] = True

            self.tiled_masses[:,annihilate] = 0
            self.particle_size[annihilate] = [0,0]

            self.tiled_masses[:,survive] += self.masses[annihilate]
            np.fill_diagonal(self.tiled_masses, 0)

            self.masses[annihilate] = 0
            print(self.tiled_masses)
            print()

        #if len(survive) != 0 or len(annihilate) != 0:
        #    print(annihilate, survive)
        #    print(self.velocity_array)
        #    print(self.masses)
        #    print(self.position_array)

        self.particle_size[:] = np.sqrt(self.masses[:,np.newaxis])*self.density

    def step(self):
        self.time += self.dt
        self.counter += 1
        self.update_directions()
        self.update_masses()
        self.update_forces()
        self.update_pos_and_velo()


        if self.counter%1000 == 0:
            print('Momentum = ', (self.masses[:, np.newaxis]*self.velocity_array).sum(axis = 0))
            print('Center of mass (can change) = ', (self.masses[:, np.newaxis]*self.position_array).sum(axis = 0))
            print()
            #print(self.exclude_mask)
