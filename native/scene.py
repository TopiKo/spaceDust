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
        self.dt = 0.001
        self.counter = 0
        self.size = Size([100,100])
        self.num_particles = 10 # num_particles
        self.collision_threshold = .5

        # Array initialization
        self.position_array = np.zeros([self.num_particles, 2])
        self.last_position_array = np.zeros([self.num_particles, 2])
        self.velocity_array = np.zeros([self.num_particles, 2])
        self.accel_array = np.zeros([self.num_particles, 2])
        self.force_array = np.zeros([self.num_particles,2])
        self.particle_size = np.zeros([self.num_particles,2])
        self.active_entries = np.ones(self.num_particles, dtype=bool)
        self.masses = np.ones(self.num_particles) #np.random.normal(loc = 1, scale = .01, size = self.num_particles, ) #


        # Masked array initialisation (why do we need the masks?)
        self.mask = np.zeros(self.num_particles).astype(bool)

        self.r_ = np.ma.array(self.position_array, mask=False)
        self.v_ = np.ma.array(self.velocity_array, mask=False)
        self.a_ = np.ma.array(self.accel_array, mask=False)
        self.F_ = np.ma.array(self.accel_array, mask=False)

        self.directions = np.ma.zeros((self.num_particles, self.num_particles, 2))

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
        self.velocity_array = np.zeros((self.num_particles, 2)) #np.cross(zs,rn)[:,:2]

        self.accel_array[:] = np.random.random(self.position_array.shape)
        self.particle_size[:] = np.sqrt(self.masses[:, np.newaxis]) #np.random.random(self.position_array.shape)+1

    def is_within_boundaries(self, coord: Point):
        """
        Check whether a single point lies within the scene.
        :param coord: Point under consideration
        :return: True if within scene, false otherwise
        """
        within_boundaries = all(np.array([0, 0]) < coord.array) and all(coord.array < self.size.array)
        return within_boundaries

    def update_directions(self):
        copy_pos = self.r_.copy()
        for i in range(self.num_particles):
            self.r_.mask[i] = True
            self.directions[i] = self.r_ - copy_pos[i]
            self.r_.mask[i] = False

    def update_forces(self):


        masses_arr = np.tile(self.masses, self.num_particles).reshape(self.directions.shape[:2] + (1,))

        dirs_m = self.directions / self.norms * masses_arr
        self.forces = np.dot(np.diag(self.masses), (dirs_m / self.norms ** 2).sum(axis=1))

    def update_pos_and_velo(self):
        """
        Performs a vectorized move of all the particles.
        Assumes that all the velocities have been set accordingly.
        :return: None
        """
        self.time += self.dt
        self.counter += 1
        self.accel_array = np.dot(np.diag(1/self.masses), self.forces)
        self.last_position_array = np.array(self.position_array)
        self.position_array += self.velocity_array * self.dt + .5*self.accel_array*self.dt**2
        self.velocity_array += self.accel_array * self.dt

    def collisions(self):
        norms = np.linalg.norm(self.directions, axis=2, keepdims=True)  # *masses

        # ---------------------
        self.norms = np.ma.array(norms, mask = self.directions.mask[:,:,:1])
        '''
        collision_idxs = np.transpose(np.where(self.norms < self.collision_threshold)).astype(int)

        firstSmaller = self.masses[collision_idxs[:,0]] < self.masses[collision_idxs[:,1]]
        annihilate_idxs = collision_idxs[firstSmaller,0]

        grow_bigger_idxs = collision_idxs[firstSmaller,1]

        # Update the masses The same is required for velocities!!
        self.masses[grow_bigger_idxs] += self.masses[annihilate_idxs]
        self.mask[annihilate_idxs] = True

        for idxs in collision_idxs[firstSmaller]:
            self.v_[idxs[1]] = (self.masses[idxs[1]]*self.v_[idxs[1]] + \
                                    self.masses[idxs[0]]*self.v_[idxs[0]])\
                                        /(self.masses[idxs[0]] + self.masses[idxs[1]])

        self.particle_size[:] = self.masses[:, np.newaxis]
        self.r_.mask[:,:] = self.mask[:,np.newaxis]
        self.v_.mask[:,:] = self.mask[:,np.newaxis]
        self.a_.mask[:,:] = self.mask[:,np.newaxis]
        self.F_.mask[:,:] = self.mask[:,np.newaxis]
        self.directions.mask[:,:,:] = self.mask[np.newaxis,:,np.newaxis]
        '''
        # ---------------------

    def step(self):
        self.time += self.dt
        self.counter += 1
        print('r_cm={}'.format((self.r_*self.masses[:,np.newaxis]).sum(axis = 0)))
        self.update_directions()
        self.collisions()
        self.update_forces()
        self.update_pos_and_velo()
