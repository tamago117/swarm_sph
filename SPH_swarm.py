import numpy as np
import matplotlib.pyplot as plt

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
        self.SMOOTHING_LENGTH = 0.8
         # [m]

        # fluid properties
        self.target_density = 0.35 # [kg/m^3]
        self.is_compressible = False
        self.VISCOSITY = 0.001 # [Pa s]
        self.TEMPERATURE = 0.001 # [K]
        self.IMCOMPRESSIVE_STIFFNESS = 0.8


        # robot properties
        self.ROBOT_RADIUS = 0.015 # [m]
        self.ROBOT_MASS = 0.05 # [kg]
        self.DT = 0.1 # [s]
        self.max_velocity = 0.1 # [m/s]

        # control variables
        self.forces = np.zeros(2)

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
    
    def calculate_pressure(self, particles):
        """
        Compute the pressure of each particle.
        """
        #pressures = np.zeros(self.NUM_PARTICLES)
        particles = self.calculate_density(particles)
        
        if self.is_compressible:
            # p = rho R(gas constant) T(temperature of water)
            pressures = [particle.density * 0.287 * self.TEMPERATURE for particle in particles]
            print(pressures)
        else:
            pressures = [self.IMCOMPRESSIVE_STIFFNESS * (particle.density - self.target_density) for particle in particles]

        for i in range(self.NUM_PARTICLES):
            particles[i].pressure = pressures[i]

        return particles
    
    def calculate_stress(self, particles):
        """
        Compute the stress tensor of each particle.
        """

        particles = self.calculate_pressure(particles)

        for particle in particles:
            #particle.stress_xx = -particle.pressure
            #particle.stress_yy = -particle.pressure
            particle.stress_xx = -particle.pressure + self.VISCOSITY * (2 * self.derivative_velocity(particle, particles, 'x') - 2/3 * (self.derivative_velocity(particle, particles, 'x') + self.derivative_velocity(particle, particles, 'y')))
            particle.stress_xy = self.VISCOSITY * (self.derivative_velocity(particle, particles, 'x') + self.derivative_velocity(particle, particles, 'y'))
            particle.stress_yy = -particle.pressure + self.VISCOSITY * (2 * self.derivative_velocity(particle, particles, 'y') - 2/3 * (self.derivative_velocity(particle, particles, 'y') + self.derivative_velocity(particle, particles, 'x')))

        return particles
    
    def calculate_velocity(self, particles):
        """
        Compute the velocity of each particle.
        """

        particles = self.calculate_stress(particles)

        for particle in particles:
            dv_x = self.forces[0]
            dv_y = self.forces[1]
            for neighbor in particles:
                r = [particle.pos_x - neighbor.pos_x, particle.pos_y - neighbor.pos_y]
                r_norm = np.linalg.norm(r)
                # calculate theta
                theta = np.arctan2(r[1], r[0])
                dv_x += self.ROBOT_MASS * ((particle.stress_xx + neighbor.stress_xx)/(particle.density*neighbor.density)*np.cos(theta) + (particle.stress_xy + neighbor.stress_xy)/(particle.density*neighbor.density)*np.sin(theta)) * self.derivative_gaussian_kernel(r_norm)
                dv_y += self.ROBOT_MASS * ((particle.stress_xy + neighbor.stress_xy)/(particle.density*neighbor.density)*np.cos(theta) + (particle.stress_yy + neighbor.stress_yy)/(particle.density*neighbor.density)*np.sin(theta)) * self.derivative_gaussian_kernel(r_norm)

                #dv_x += self.ROBOT_MASS * ((particle.stress_xx + neighbor.stress_xx)/(particle.density*neighbor.density) + (particle.stress_xy + neighbor.stress_xy)/(particle.density*neighbor.density)) * self.derivative_gaussian_kernel(r_norm)
                #dv_y += self.ROBOT_MASS * ((particle.stress_xy + neighbor.stress_xy)/(particle.density*neighbor.density) + (particle.stress_yy + neighbor.stress_yy)/(particle.density*neighbor.density)) * self.derivative_gaussian_kernel(r_norm)

            #摩擦項
            dv_x += -particle.vel_x * 0.35
            dv_y += -particle.vel_y * 0.35
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


if __name__ == "__main__":
    swarm = sph_swarm()
    particles = [sph_particle() for i in range(swarm.NUM_PARTICLES)]

    way_points = [[0.3, 0.3], [0.3, -0.3], [-0.3, -0.3], [-0.3, 0.3], [0.0, 0.0]]
    way_number = 0

    for i in range(swarm.NUM_PARTICLES):
        particles[i].pos_x = np.random.uniform(-0.1, 0.1)
        particles[i].pos_y = np.random.uniform(-0.1, 0.1)
        #particles[i].vel_x = np.random.uniform(0, 1)
        #particles[i].vel_y = np.random.uniform(0, 1)

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
        force_x = -(ave_x - way_points[way_number][0]) * 0.1
        force_y = -(ave_y - way_points[way_number][1]) * 0.1

        swarm.set_forces([force_x, force_y])

        swarm.calculate_velocity(particles)
        for particle in particles:
            particle.pos_x += particle.vel_x * swarm.DT
            particle.pos_y += particle.vel_y * swarm.DT

    # plot        
        plt.clf()
        plt.xlim(-0.8, 0.8)
        plt.ylim(-0.8, 0.8)
        plt.scatter([particle.pos_x for particle in particles], [particle.pos_y for particle in particles])
        # velocity vector
        plt.quiver([particle.pos_x for particle in particles], [particle.pos_y for particle in particles], [particle.vel_x for particle in particles], [particle.vel_y for particle in particles], angles='xy', scale_units='xy', scale=3)
        # all way points
        plt.scatter([way_point[0] for way_point in way_points], [way_point[1] for way_point in way_points])
        plt.pause(0.01)

        # next way point
        if np.linalg.norm([ave_x - way_points[way_number][0], ave_y - way_points[way_number][1]]) < 0.05:
            way_number += 1
            if way_number == len(way_points):
                break

        

        # keyboard interrupt
        if plt.waitforbuttonpress(0.01):
            break
         