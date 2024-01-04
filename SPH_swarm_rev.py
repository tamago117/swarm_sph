import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from pointInside import *

class sph_particle:
    def __init__(self):
        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 0
        self.vel_y = 0
        self.density = 0
        self.pressure = 0
        self.stress_xx = 0
        self.stress_xy = 0
        self.stress_yy = 0

class sph_swarm:
    def __init__(self):
        # sph parameters
        self.NUM_PARTICLES = 15
        self.SMOOTHING_LENGTH = 0.8 # [m]

        # fluid properties
        self.target_density = 0.36 # [kg/m^3]
        self.is_compressible = False
        self.VISCOSITY = 0.0001 # [Pa s]
        self.TEMPERATURE = 0.001 # [K]
        self.IMCOMPRESSIVE_STIFFNESS = 0.8

        # robot properties
        self.ROBOT_RADIUS = 0.015 # [m]
        self.ROBOT_MASS = 0.05 # [kg]
        self.DT = 0.1 # [s]
        self.max_velocity = 0.1 # [m/s]

        # control variables
        self.forces = np.zeros(2)

        # swarm form ((x1, y1), (x2, y2), ...)
        self.swarm_shape = np.zeros((1, 2))

    def gauss_kernel(self, radius : float):
        """
        Compute the Gaussian kernel.
        """
        a = 1 / (np.pi * self.SMOOTHING_LENGTH**2)
        b = radius/self.SMOOTHING_LENGTH
        k = 2

        if b <= k*self.SMOOTHING_LENGTH:
            return a * np.exp(-b**2)
        else:
            return 0.0001
        
    def derivative_gaussian_kernel(self, radius : float):
        """
        Compute the derivative of Gaussian kernel.
        """
        a = 1 / (np.pi * self.SMOOTHING_LENGTH**2)
        b = radius/self.SMOOTHING_LENGTH
        k = 2

        if b <= k*self.SMOOTHING_LENGTH:
            return -2/(self.SMOOTHING_LENGTH**2) * b * a * np.exp(-b**2)
        else:
            return 0
        
    def derivative_velocity(self, particle, particles, dir : str):
        """
        Compute the derivative of SPH kernel.
        """

        result = 0

        # find neighbors( r < h)

        
        for neighbor in particles:
            r = [particle.pos_x - neighbor.pos_x, particle.pos_y - neighbor.pos_y]
            r_norm = np.linalg.norm(r)
            if dir == 'x':
                result += neighbor.vel_x * self.derivative_gaussian_kernel(r_norm)
            elif dir == 'y':
                result += neighbor.vel_y * self.derivative_gaussian_kernel(r_norm)

        return result

    def calculate_density(self, particles):
        """
        Compute the density of each particle
        """

        for particle in particles:
            particle.density = 0
            for neighbor in particles:
                r = [particle.pos_x - neighbor.pos_x, particle.pos_y - neighbor.pos_y]
                r_norm = np.linalg.norm(r)
                particle.density += self.ROBOT_MASS * self.gauss_kernel(r_norm)
                
        return particles
    
    def artificial_viscosity(self, particle, neighbor):
        B = 10
        psi = 0.1*self.SMOOTHING_LENGTH
        density_pn = 0.5*(particle.density + neighbor.density)

        v_p2n = [particle.vel_x - neighbor.vel_x, particle.vel_y - neighbor.vel_y]
        x_p2n = [particle.pos_x - neighbor.pos_x, particle.pos_y - neighbor.pos_y]
        # vとxの内積
        v_dot_x = v_p2n[0]*x_p2n[0] + v_p2n[1]*x_p2n[1]

        if v_dot_x < 0:
            phi = self.SMOOTHING_LENGTH*v_dot_x/(np.linalg.norm(x_p2n)**2 + psi**2)
            result = B*phi**2 / density_pn
            return result
        else:
            return 0
    
    def calculate_pressure(self, particles):
        """
        Compute the pressure of each particle.
        """
        gamma = 7

        #pressures = np.zeros(self.NUM_PARTICLES)
        particles = self.calculate_density(particles)
        
        if self.is_compressible:
            # p = rho R(gas constant) T(temperature of water)
            pressures = [particle.density * 0.287 * self.TEMPERATURE for particle in particles]
        else:
            pressures = [self.IMCOMPRESSIVE_STIFFNESS * ((particle.density/self.target_density)**gamma - 1) for particle in particles]

        for i in range(self.NUM_PARTICLES):
            particles[i].pressure = pressures[i]

        return particles
    
    def calculate_stress(self, particles):
        """
        Compute the stress tensor of each particle.
        """

        particles = self.calculate_pressure(particles)

        for particle in particles:
            particle.stress_xx = 0
            particle.stress_xy = 0
            particle.stress_yy = 0
            for neighbor in particles:
                r = [particle.pos_x - neighbor.pos_x, particle.pos_y - neighbor.pos_y]
                r_norm = np.linalg.norm(r)
                theta = np.arctan2(r[1], r[0])
                vel_x_n2p = neighbor.vel_x - particle.vel_x
                vel_y_n2p = neighbor.vel_y - particle.vel_y

                particle.stress_xx += 2/3*(self.ROBOT_MASS/neighbor.density) * (2*vel_x_n2p*self.derivative_gaussian_kernel(r_norm)*np.cos(theta) - vel_y_n2p*self.derivative_gaussian_kernel(r_norm)*np.sin(theta))
                particle.stress_xy += (self.ROBOT_MASS/neighbor.density) * (vel_y_n2p*self.derivative_gaussian_kernel(r_norm)*np.cos(theta) + vel_x_n2p*self.derivative_gaussian_kernel(r_norm)*np.sin(theta))
                particle.stress_yy += 2/3*(self.ROBOT_MASS/neighbor.density) * (2*vel_y_n2p*self.derivative_gaussian_kernel(r_norm)*np.sin(theta) - vel_x_n2p*self.derivative_gaussian_kernel(r_norm)*np.cos(theta))
    
        return particles
    
    def calculate_sph_force(self, particle, neighbors):
        force_x = 0
        force_y = 0

        for neighbor in neighbors:
            if particle == neighbor:
                continue
            r = [particle.pos_x - neighbor.pos_x, particle.pos_y - neighbor.pos_y]
            r_norm = np.linalg.norm(r)
            # calculate theta
            theta = np.arctan2(r[1], r[0])

            force_x += -self.ROBOT_MASS * ((particle.pressure + neighbor.pressure)/(particle.density*neighbor.density) + self.artificial_viscosity(particle, neighbor)) * self.derivative_gaussian_kernel(r_norm) * np.cos(theta) \
                       + self.ROBOT_MASS*self.VISCOSITY/(particle.density*neighbor.density)*((particle.stress_xx + neighbor.stress_xx)*self.derivative_gaussian_kernel(r_norm)*np.cos(theta) + (particle.stress_xy + neighbor.stress_xy)*self.derivative_gaussian_kernel(r_norm)*np.sin(theta))
            force_y += -self.ROBOT_MASS * ((particle.pressure + neighbor.pressure)/(particle.density*neighbor.density) + self.artificial_viscosity(particle, neighbor)) * self.derivative_gaussian_kernel(r_norm) * np.sin(theta) \
                       + self.ROBOT_MASS*self.VISCOSITY/(particle.density*neighbor.density)*((particle.stress_xy + neighbor.stress_xy)*self.derivative_gaussian_kernel(r_norm)*np.cos(theta) + (particle.stress_yy + neighbor.stress_yy)*self.derivative_gaussian_kernel(r_norm)*np.sin(theta))

        return [force_x, force_y]
    
    def calculate_potential_field(self, particle, neighbors):
        force_x = 0
        force_y = 0

        for neighbor in neighbors:
            if particle == neighbor:
                continue
            r = [particle.pos_x - neighbor.pos_x, particle.pos_y - neighbor.pos_y]
            r_norm = np.linalg.norm(r)

            EPSILON = 1e-8
            force_x += r[0] / (r_norm + EPSILON)**2 * self.gauss_kernel(r_norm)
            force_y += r[1] / (r_norm + EPSILON)**2 * self.gauss_kernel(r_norm)            

        return [force_x, force_y]
    
    def caluculate_swarm_shape_force(self, particle, particles):
        """
        Compute the force from swarm shape.
        """

        force_x = 0
        force_y = 0

        NEAR_POINTS = 3

        # KDTree
        tree = KDTree(self.swarm_shape)
        dist, ind = tree.query([[particle.pos_x, particle.pos_y]], k=NEAR_POINTS)
        # serch nearest points
        nearest_points = self.swarm_shape[ind[0][0:NEAR_POINTS]]

        # area interior/exterior judgment (judgement)
        if is_inside_sm(self.swarm_shape, [particle.pos_x, particle.pos_y]) == 1:
            # inside
            return [0, 0]

        # calculate force
        for i in range(NEAR_POINTS):
            if NEAR_POINTS == 1:
                r = [particle.pos_x - nearest_points[0], particle.pos_y - nearest_points[1]]
            else:
                r = [particle.pos_x - nearest_points[i][0], particle.pos_y - nearest_points[i][1]]
            r_norm = np.linalg.norm(r)

            # suction force
            force_x += -r[0] * self.gauss_kernel(r_norm)
            force_y += -r[1] * self.gauss_kernel(r_norm)

        force_x /= NEAR_POINTS
        force_y /= NEAR_POINTS

        return [force_x, force_y]

    def calculate_velocity(self, particles):
        """
        Compute the velocity of each particle.
        """

        particles = self.calculate_stress(particles)

        for particle in particles:
            # sph_force
            sph_force = self.calculate_sph_force(particle, particles)

            # potential field
            potential_field = self.calculate_potential_field(particle, particles)

            # swarm shape
            swarm_shape_force = self.caluculate_swarm_shape_force(particle, particles)

            # control force + sph_force + potential_field + swarm_shape_force - friction
            dv_x = self.forces[0] + sph_force[0] + 0.005*potential_field[0] + 45*swarm_shape_force[0] - 0.35*particle.vel_x
            dv_y = self.forces[1] + sph_force[1] + 0.005*potential_field[1] + 45*swarm_shape_force[1] - 0.35*particle.vel_y

            # update velocity
            particle.vel_x = particle.vel_x + dv_x*self.DT
            particle.vel_y = particle.vel_y + dv_y*self.DT

            # max velocity
            if np.linalg.norm([particle.vel_x, particle.vel_y]) > self.max_velocity:
                particle.vel_x = particle.vel_x / np.linalg.norm([particle.vel_x, particle.vel_y]) * self.max_velocity
                particle.vel_y = particle.vel_y / np.linalg.norm([particle.vel_x, particle.vel_y]) * self.max_velocity

        return particles
 
    def set_target_density(self, target_density):
        self.target_density = target_density

    def set_compressible(self, is_compressible):
        self.is_compressible = is_compressible

    def set_viscosity(self, viscosity):
        self.VISCOSITY = viscosity

    def set_temperature(self, temperature):
        self.TEMPERATURE = temperature

    def set_imcompressive_stiffness(self, imcompressive_stiffness):
        self.IMCOMPRESSIVE_STIFFNESS = imcompressive_stiffness

    def set_forces(self, forces):
        self.forces = forces

    def set_swarm_shape(self, swarm_shape : np.ndarray):
        self.swarm_shape = swarm_shape


if __name__ == "__main__":
    swarm = sph_swarm()
    particles = [sph_particle() for i in range(swarm.NUM_PARTICLES)]

    way_points = [[0.3, 0.3], [0.3, -0.3], [-0.3, -0.3], [-0.3, 0.3], [0.0, 0.0]]
    way_number = 0

    # initialize
    for i in range(swarm.NUM_PARTICLES):
        #　列に並べる
        particles[i].pos_x = 0.0
        particles[i].pos_y = -0.5 + 0.1*i
        
        # random
        #particles[i].pos_x = np.random.uniform(-0.1, 0.1)
        #particles[i].pos_y = np.random.uniform(-0.1, 0.1)
        #particles[i].vel_x = np.random.uniform(0, 1)
        #particles[i].vel_y = np.random.uniform(0, 1)

    # swarm form
    # 四角形を各辺10個の点で表現
    swarm_shape = np.zeros((40, 2))
    for i in range(10):
        swarm_shape[i][0] = 0.05
        swarm_shape[i][1] = 0.15 - 0.03*i
    for i in range(10):
        swarm_shape[i+10][0] = 0.05 - 0.01*i
        swarm_shape[i+10][1] = -0.15
    for i in range(10):
        swarm_shape[i+20][0] = -0.05
        swarm_shape[i+20][1] = -0.15 + 0.03*i
    for i in range(10):
        swarm_shape[i+30][0] = -0.05 + 0.01*i
        swarm_shape[i+30][1] = 0.15
    
   
    
    swarm.set_swarm_shape(swarm_shape)

    # plot configuration
    plt.gca().set_aspect('equal', adjustable='box')

    # simulation loop
    while True:
        # average point
        ave_x = 0
        ave_y = 0
        ave_density = 0
        ave_pressure = 0
        for particle in particles:
            ave_x += particle.pos_x
            ave_y += particle.pos_y
            ave_density += particle.density
            ave_pressure += particle.pressure
        ave_x /= swarm.NUM_PARTICLES
        ave_y /= swarm.NUM_PARTICLES
        ave_density /= swarm.NUM_PARTICLES
        ave_pressure /= swarm.NUM_PARTICLES
        print("pos x:", round(ave_x, 3), "pos y:", round(ave_y, 3), "density:", round(ave_density, 5), "pressure:", round(ave_pressure, 5))
        
        # controller
        force_x = -(ave_x - way_points[way_number][0]) * 0.2
        force_y = -(ave_y - way_points[way_number][1]) * 0.2

        #swarm.set_forces([force_x, force_y])

        swarm.calculate_velocity(particles)
        for particle in particles:
            particle.pos_x += particle.vel_x * swarm.DT
            particle.pos_y += particle.vel_y * swarm.DT

        # plot
        # save video        
        plt.clf()
        plt.xlim(-0.8, 0.8)
        plt.ylim(-0.8, 0.8)
        plt.scatter([particle.pos_x for particle in particles], [particle.pos_y for particle in particles])
        # velocity vector
        plt.quiver([particle.pos_x for particle in particles], [particle.pos_y for particle in particles], [particle.vel_x for particle in particles], [particle.vel_y for particle in particles], angles='xy', scale_units='xy', scale=3)
        # all way points
        plt.scatter([way_point[0] for way_point in way_points], [way_point[1] for way_point in way_points])
        # swarm shape
        plt.scatter(swarm_shape[:, 0], swarm_shape[:, 1], color='red', s=1)
        plt.pause(0.01)

        

        # next way point
        if np.linalg.norm([ave_x - way_points[way_number][0], ave_y - way_points[way_number][1]]) < 0.01:
            way_number += 1
            if way_number == len(way_points):
                break

        # keyboard interrupt
        if plt.waitforbuttonpress(0.01):
            break
         
         
