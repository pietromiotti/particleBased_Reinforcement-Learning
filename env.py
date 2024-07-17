import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as rand
from numpy import linalg as LA
import torch.nn as nn
import gym

import torch
#Remember that 'actorcriticcontinuous have been trained with lenght wall up =4'
class Env(gym.Env):
    def __init__(self, m=80,  L=10, lenghtWall_up=2, lenghtWall_bottom=0, obstacleLenght = 0,  init_n_particles=5000, dt=0.01, continuous=True):

        #continuous
        self.continuous = continuous
        if(self.continuous):
            self.num_action = 1
        else:
            self.num_action = 8

        self.num_state = 4
        self.selfDrivenForce = 8
        self.rewardSuccess = 0
        self.rewardFailed = -1

        self.rc = 0.5
        self.entranceTrajectoryPenalty = 0.1
        self.di = 0.01
        self.iterationPerEpisode = 5000

        self.episodes_ppo = 4000
        self.episodes_dqn = 3000
        self.m = m
        self.L = L
        self.lenghtWall_up = lenghtWall_up
        self.lenghtWall_bottom = lenghtWall_bottom

        self.number_of_fluid_particles = 1

        self.sins = [0, np.sqrt(2) / 2, 1, np.sqrt(2) / 2, 0, -np.sqrt(2) / 2, -1, -np.sqrt(2) / 2]
        self.coss = [1, np.sqrt(2) / 2, 0, -np.sqrt(2) / 2, -1, -np.sqrt(2) / 2, 0, np.sqrt(2) / 2]
        self.WALL_LEFT = 1
        self.WALL_RIGHT = L-2
        self.WALL_UP = L-2
        self.WALL_BOTTOM = 1
        self.pos_up_wall = np.array([(self.L-1)/2, self.WALL_UP])
        self.pos_bottom_wall = np.array([(self.L-1)/2,self.WALL_BOTTOM])


        self.radiuswall_up = self.lenghtWall_up/2
        self.radiuswall_bottom = self.lenghtWall_bottom/2

        self.startWall_up = (self.L-1) / 2 - self.lenghtWall_up / 2
        self.endWall_up = (self.L-1) / 2 + self.lenghtWall_up / 2
        self.startWall_bottom = (self.L-1) / 2 - self.lenghtWall_bottom / 2
        self.endWall_bottom = (self.L-1) / 2 + self.lenghtWall_bottom / 2


        self.startObstacleX = (self.L - 1) / 2 - obstacleLenght/2
        self.endObstacleX= (self.L - 1) / 2 + obstacleLenght/2
        self.endObstacleY= (self.L -1)/2 + obstacleLenght/2
        self.startObstacleY= (self.L -1)/2 - obstacleLenght/2

        self.N_particles = init_n_particles
        self.pos = (L-1) * np.random.rand(2, self.N_particles)

        self.wall_left_p = self.pos[0,:] < self.WALL_LEFT
        self.wall_right_p = self.pos[0,:] > self.WALL_RIGHT

        self.wall_up_p = (self.pos[1,:] > self.WALL_UP) \
                         & (np.logical_not((self.pos[0,:] < self.WALL_LEFT)
                                         | (self.pos[0,:] > self.WALL_RIGHT)))
        self.wall_bottom_p = (self.pos[1,:] < self.WALL_BOTTOM) \
                             & (np.logical_not((self.pos[0, :] < self.WALL_LEFT)
                                             | (self.pos[0, :] > self.WALL_RIGHT)))


        self.obstacle_p_particle = (self.pos[1,:]>self.startObstacleY) & (self.pos[1,:]<self.endObstacleY) & (self.pos[0,:] > self.startObstacleX) & (self.pos[0,:] < self.endObstacleX)

        self.wall_bottom_p_entrance = (self.wall_bottom_p & (self.pos[0,:] > self.startWall_bottom) & (self.pos[0,:] < self.endWall_bottom)) | (self.wall_up_p & (self.pos[0,:] > self.startWall_up) & (self.pos[0,:] < self.endWall_up))

        self.wall_bottom_entrance_particles = np.where(self.wall_bottom_p_entrance == 1)


        self.obstacle_particle = np.where(self.obstacle_p_particle == 1)[0]
        self.left_wall_particles = np.where(self.wall_left_p == 1)[0]
        self.right_wall_particles = np.where(self.wall_right_p == 1)[0]
        self.bottom_wall_particles = np.where(self.wall_bottom_p == 1)[0]
        self.up_wall_particles = np.where(self.wall_up_p == 1)[0]

        self.total_wall_p = np.sum(self.wall_left_p) + np.sum(self.wall_right_p) + np.sum(self.wall_bottom_p) + np.sum(
            self.wall_up_p) + np.sum(self.obstacle_particle)


        self.removeElementsFromWallList(self.wall_bottom_entrance_particles[0])

        self.availableParticles_samplingSet = np.ones(shape=(1,self.N_particles), dtype='bool') & (np.logical_not(self.wall_left_p)) & (np.logical_not(self.wall_right_p)) & (np.logical_not(self.wall_up_p)) & (np.logical_not(self.wall_bottom_p)) & (np.logical_not(self.obstacle_p_particle))

        self.indices_availableParticles = np.where(self.availableParticles_samplingSet == 1)[1]

        self.indicesFluidParticles = np.random.choice(self.indices_availableParticles, size=self.number_of_fluid_particles)
        self.particlesToDelete = np.setdiff1d(self.indices_availableParticles, self.indicesFluidParticles)

        #remove non used particles
        self.removeElementsFromWallList(self.particlesToDelete)

        #init the map
        self.initMap(L,L)

        self.fluid_particles = np.ones(shape=(1,self.N_particles), dtype='bool') & (np.logical_not(self.wall_left_p)) & (np.logical_not(self.wall_right_p)) & (np.logical_not(self.wall_up_p)) & (np.logical_not(self.wall_bottom_p)) & (np.logical_not(self.obstacle_p_particle))

        self.fluid_particles_id = np.where(self.fluid_particles==1)[1]

        self.TYPE_WALL = 2
        self.TYPE_FLUID = 1

        self.total_particles = self.TYPE_WALL * self.wall_left_p + self.TYPE_WALL * self.wall_right_p + self.TYPE_WALL * self.wall_up_p + self.TYPE_WALL * self.wall_bottom_p + self.TYPE_FLUID * self.fluid_particles + self.TYPE_WALL * self.obstacle_p_particle
        self.total_particles= self.total_particles[0]

        #init velocities
        self.Vx = np.zeros(shape=(1,self.N_particles))
        self.Vy = np.zeros(shape=(1,self.N_particles))

        #adjust velocities in order to make the system isolated
        if (np.sum(self.Vx + self.Vy) !=0):
            delta = np.sum(self.Vx + self.Vy) / (2 * self.N_particles)
            self.Vx = self.Vx - delta
            self.Vy = self.Vy - delta


        self.V = np.vstack((self.Vx, self.Vy))

        #set all velocities to 0
        self.resetWallVelocities()


        self.dt = dt
        self.iter = 0

        self.initialStatePos = self.pos
        self.initialStateVel = self.V
        self.initialStateTotalParticles = self.total_particles

        self.cell_particles = np.zeros(shape=(self.N_particles,1))

        self.computeCellParticlesAll()

        total_forces = np.zeros(shape=(2, self.N_particles))


        for particle in self.fluid_particles_id:

            rij, entrance = self.computeRandEntrance(particle)

            if (rij < self.rc) and not (entrance):
                cellId = self.computeCell(self.pos[:, particle])

                for q in range(9):
                    neighbour_cell = self.neighbours[cellId,q]

                    neighbour_particles_all = np.where(self.cell_particles == neighbour_cell)
                    for p in range(len(neighbour_particles_all[0])):
                        neighbour_particles_single = neighbour_particles_all[0][p]

                        rij_vec_x = self.pos[0, particle] - self.pos[0, neighbour_particles_single]
                        rij_vec_y = self.pos[1, particle] - self.pos[1, neighbour_particles_single]

                        rij = np.sqrt(rij_vec_x**2 + rij_vec_y**2)

                        if (rij < self.rc) and (neighbour_particles_single != particle):

                            rij_hat_x = rij_vec_x / rij
                            rij_hat_y = rij_vec_y / rij

                            tij_vec_x = -rij_vec_y
                            tij_vec_y = rij_vec_x
                            tij_hat_x = - rij_hat_y
                            tij_hat_y = rij_hat_x

                            dij = (self.di+self.di)/2

                            [forceAx, forceAy] = self.avoidance_coordinates(rij, dij, rij_hat_x, rij_hat_y)

                            vij_x = self.V[0, particle] - 0
                            vij_y = self.V[1, particle] - 0

                            #not used
                            uij_hat_x = vij_x * tij_hat_x
                            uij_hat_y = vij_y * tij_hat_y

                            [forceCx, forceCy] = self.compression_coordinates(rij, dij, rij_hat_x, rij_hat_y)

                            [forceFx, forceFy] = self.friction_coordinates(rij, dij, tij_hat_x, tij_hat_y, tij_vec_x,tij_vec_y, vij_x, vij_y)

                            total_forces[0, particle] = total_forces[0, particle] + forceAx + forceCx + forceFx
                            total_forces[1, particle] = total_forces[1, particle] + forceAy + forceCy + forceFy

            [forceVx, forceVy] = self.viscous_force(self.V[0, particle], self.V[1, particle], dt, self.m)

            total_forces[0, particle] = total_forces[0, particle] + forceVx
            total_forces[1, particle] = total_forces[1, particle] + forceVy

        self.acc = np.zeros(shape=(2,self.N_particles))
        self.acc[0,:] = total_forces[0,:]/ self.m
        self.acc[1,:] = total_forces[1,:]/ self.m

    def computeRandEntrance(self, particle):
        rij_vec_x_left = abs(self.pos[0, particle] - self.WALL_LEFT)
        rij_vec_x_right = abs(self.pos[0, particle] - self.WALL_RIGHT)

        if (rij_vec_x_left < rij_vec_x_right):
            rij_vec_x = self.pos[0, particle] - self.WALL_LEFT
        else:
            rij_vec_x = self.pos[0, particle] - self.WALL_RIGHT

        rij_vec_y_top = abs(self.pos[1, particle] - self.WALL_UP)
        rij_vec_y_bottom = abs(self.pos[1, particle] - self.WALL_BOTTOM)

        rij_vec_y_obstacle_bottom = abs(self.pos[1, particle] - self.startObstacleY)
        rij_vec_y_obstacle_top = abs(self.pos[1, particle] - self.endObstacleY)

        rij_vec_x_obstacle_left = abs(self.pos[0, particle] - self.startObstacleX)
        rij_vec_x_obstacle_right = abs(self.pos[0, particle] - self.endObstacleX)

        rij_vec_y_obstacle = min(rij_vec_y_obstacle_bottom, rij_vec_y_obstacle_top)
        rij_vec_x_obstacle = min(rij_vec_x_obstacle_left, rij_vec_x_obstacle_right)

        if (rij_vec_y_top < rij_vec_y_bottom):
            rij_vec_y = self.pos[1, particle] - self.WALL_UP
        else:
            rij_vec_y = self.pos[1, particle] - self.WALL_BOTTOM

        rij = np.sqrt(rij_vec_x ** 2 + rij_vec_y ** 2)

        rij = min(rij, rij_vec_x_left, rij_vec_x_right, rij_vec_y_top, rij_vec_y_bottom, rij_vec_y_obstacle,
                  rij_vec_x_obstacle)

        dist_up = LA.norm(self.pos[:, particle] - self.pos_up_wall)
        dist_bottom = LA.norm(self.pos[:, particle] - self.pos_bottom_wall)

        entrance = (dist_up < (self.radiuswall_up - self.entranceTrajectoryPenalty)) or (
                    dist_bottom < (self.radiuswall_bottom - self.entranceTrajectoryPenalty))

        return rij, entrance

    def computeCellParticlesAll(self):
        for i in range(self.N_particles):
            self.cell_particles[i] = self.computeCell(self.pos[:, i])



    def removeElementsFromWallList(self, index):

        self.obstacle_p_particle = np.delete(self.obstacle_p_particle,index)
        self.wall_bottom_p = np.delete(self.wall_bottom_p, index)
        self.wall_up_p = np.delete(self.wall_up_p, index)
        self.wall_left_p = np.delete(self.wall_left_p, index)
        self.wall_right_p = np.delete(self.wall_right_p, index)


        self.pos = np.delete(self.pos, index, axis=1)


        self.N_particles = self.N_particles - len(index)

        self.left_wall_particles = np.where(self.wall_left_p == 1)[0]
        self.right_wall_particles = np.where(self.wall_right_p == 1)[0]
        self.bottom_wall_particles = np.where(self.wall_bottom_p == 1)[0]
        self.up_wall_particles = np.where(self.wall_up_p == 1)[0]
        self.obstacle_particle = np.where(self.obstacle_p_particle==1)[0]

        self.total_wall_p = np.sum(self.wall_left_p) + np.sum(self.wall_right_p) + np.sum(self.wall_bottom_p) + np.sum(self.wall_up_p) + np.sum(self.obstacle_p_particle)


    def entranceTrajectory(self, particle):
        dist_up = LA.norm(self.pos[:, particle]-self.pos_up_wall)
        dist_bottom = LA.norm(self.pos[:, particle]-self.pos_bottom_wall)
        bool2 =  (dist_up < self.radiuswall_up) or (dist_bottom < self.radiuswall_bottom)

        bool_ = ((self.pos[0, particle] > self.startWall_up +self.entranceTrajectoryPenalty) and (
                    self.pos[0, particle] < self.endWall_up - self.entranceTrajectoryPenalty) and (self.pos[1, particle]>self.L/2)) or ((self.pos[0, particle] > self.startWall_bottom +self.entranceTrajectoryPenalty) and (
                    self.pos[0, particle] < self.endWall_bottom - self.entranceTrajectoryPenalty) and (self.pos[1, particle] < self.L/2))
        return bool_

    def initMap(self, rows, columns):

        self.rows = rows
        self.columns = columns
        self.map = np.ndarray(shape=(rows, columns), dtype=int)
        self.size = rows * columns
        self.cells = {}
        self.coords = np.ndarray(shape=(rows * columns, 2), dtype=int)
        self.neighbours = np.ndarray(shape=(rows * columns, 9), dtype=int)

        for i in range(0, rows):
            for j in range(0, columns):
                id = i * rows + j
                self.cells[id] = []
                self.map[i, j] = id
                self.coords[id] = [i, j]

        for key in self.cells:
            X = self.coords[key][0]
            Y = self.coords[key][1]
            e = self.map[X, (Y + 1) % self.columns]
            w = self.map[X, (Y - 1) % self.columns]
            s = self.map[(X + 1) % self.rows, Y]
            n = self.map[(X - 1) % self.rows, Y]
            se = self.map[(X + 1) % self.rows, (Y + 1) % self.columns]
            sw = self.map[(X + 1) % self.rows, (Y - 1) % self.columns]
            ne = self.map[(X - 1) % self.rows, (Y + 1) % self.columns]
            nw = self.map[(X - 1) % self.rows, (Y - 1) % self.columns]
            self.neighbours[key] = np.array([e, w, n, s, se, sw, ne, nw, self.map[X,Y]])


    def computeCell(self, pos):
        x = int(np.floor(pos[0]))
        y = int(np.floor(pos[1]))
        cell = self.map[x, y]
        return cell

    def moveParticleCell(self, particleId):
        self.cell_particles[particleId] = self.computeCell(self.pos[:, particleId])


    def resetWallVelocities(self):

        for i in range(len(self.left_wall_particles)):
            self.V[0, self.left_wall_particles[i]] = 0
            self.V[1, self.left_wall_particles[i]] = 0

        for i in range(len(self.right_wall_particles)):
            self.V[0, self.right_wall_particles[i]] = 0
            self.V[1, self.right_wall_particles[i]] = 0

        for i in range(len(self.bottom_wall_particles)):
            self.V[0, self.bottom_wall_particles[i]] = 0
            self.V[1, self.bottom_wall_particles[i]] = 0

        for i in range(len(self.up_wall_particles)):
            self.V[0, self.up_wall_particles[i]] = 0
            self.V[1, self.up_wall_particles[i]] = 0


    def compression_coordinates(self, rij, dij, rij_hat_x, rij_hat_y,):
        forceCx = 4.0*1e4*np.heaviside((rij-dij),0)*rij_hat_x
        forceCy = 4.0*1e4*np.heaviside((rij-dij),0)*rij_hat_y
        return [forceCx, forceCy]

    def avoidance_coordinates(self, rij, dij, rij_hat_x, rij_hat_y, A=100, b=0.08):
        forceAx = A * np.exp((dij - rij)/b) * rij_hat_x
        forceAy = A * np.exp((dij - rij)/b) * rij_hat_y
        return [forceAx,forceAy]

    def friction_coordinates(self,rij, dij, tij_hat_x, tij_hat_y, tij_vec_x,tij_vec_y, vix, viy):
        forceFx = 4.0*1e4*np.heaviside((dij-rij),0)*vix*tij_vec_x* tij_hat_x
        forceFy = 4.0*1e4*np.heaviside((dij-rij), 0)*viy*tij_vec_y * tij_hat_y
        return [forceFx, forceFy]

    def viscous_force(self, vx, vy, dt, m):
        forceVx = - (m*vx)/(dt)
        forceVy = - (m*vy)/(dt)
        return [forceVx, forceVy]

    def reset(self, maxendpoint=0):

        #get a new random state for fluid particle
        for particle in self.fluid_particles_id:
            bool_val = True
            while(bool_val):
                self.pos[0,particle] = rand.uniform(2, self.L-2)
                #self.pos[1,particle] = rand.uniform(2, self.L-3)
                if(maxendpoint>0):
                    self.pos[1, particle] = rand.uniform(2, maxendpoint)
                else:
                    self.pos[1, particle] = rand.uniform(2, self.L-2)

                bool_val = (self.pos[1,particle]>self.startObstacleY) & (self.pos[1,particle]<self.endObstacleY) & (self.pos[0,particle] > self.startObstacleX) & (self.pos[0,particle] < self.endObstacleX)
            self.moveParticleCell(particle)

        self.Vx = np.zeros(shape=(1,self.N_particles))
        self.Vy = np.zeros(shape=(1,self.N_particles))

        # adjust velocities in order to make the system isolated
        if (np.sum(self.Vx + self.Vy) != 0):
            delta = np.sum(self.Vx + self.Vy) / (2 * self.N_particles)
            self.Vx = self.Vx - delta
            self.Vy = self.Vy - delta

        self.V = np.vstack((self.Vx, self.Vy))
        # set all velocities to 0
        self.resetWallVelocities()

        self.iter = 0


        #self.computeCellParticlesAll()

        total_forces = np.zeros(shape=(2, self.N_particles))

        particleID = 0
        for particle in self.fluid_particles_id:
            particleID = particle

            rij, entrance = self.computeRandEntrance(particle)

            #entrance = ((self.pos[0, particle] > self.startWall_up + self.entranceTrajectoryPenalty) and (
            #       self.pos[0, particle] < self.endWall_up - self.entranceTrajectoryPenalty) and  (self.pos[1, particle]>self.L/2)) or ((self.pos[0, particle] > self.startWall_bottom + self.entranceTrajectoryPenalty) and (
            #        self.pos[0, particle] < self.endWall_bottom - self.entranceTrajectoryPenalty) and (self.pos[1, particle] < self.L/2))

            if (rij < self.rc) and not (entrance):
                cellId = self.computeCell(self.pos[:, particle])

                for q in range(9):
                    neighbour_cell = self.neighbours[cellId, q]

                    neighbour_particles_all = np.where(self.cell_particles == neighbour_cell)

                    for p in range(len(neighbour_particles_all[0])):
                        neighbour_particles_single = neighbour_particles_all[0][p]

                        rij_vec_x = self.pos[0, particle] - self.pos[0, neighbour_particles_single]
                        rij_vec_y = self.pos[1, particle] - self.pos[1, neighbour_particles_single]

                        rij = np.sqrt(rij_vec_x ** 2 + rij_vec_y ** 2)

                        if (rij < self.rc) and (neighbour_particles_single != particle):
                            type_neighbour = self.TYPE_WALL
                            rij_hat_x = rij_vec_x / rij
                            rij_hat_y = rij_vec_y / rij

                            tij_vec_x = -rij_vec_y
                            tij_vec_y = rij_vec_x
                            tij_hat_x = - rij_hat_y
                            tij_hat_y = rij_hat_x

                            dij = (self.di + self.di) / 2

                            [forceAx, forceAy] = self.avoidance_coordinates(rij, dij, rij_hat_x, rij_hat_y)

                            vij_x = self.V[0, particle] - 0
                            vij_y = self.V[1, particle] - 0

                            #not used
                            uij_hat_x = vij_x * tij_hat_x
                            uij_hat_y = vij_y * tij_hat_y

                            [forceCx, forceCy] = self.compression_coordinates(rij, dij, rij_hat_x, rij_hat_y)

                            [forceFx, forceFy] = self.friction_coordinates(rij, dij, tij_hat_x, tij_hat_y, tij_vec_x,
                                                                           tij_vec_y, vij_x, vij_y)

                            total_forces[0, particle] = total_forces[0, particle] + forceAx + forceCx + forceFx
                            total_forces[1, particle] = total_forces[1, particle] + forceAy + forceCy + forceFy

            [forceVx, forceVy] = self.viscous_force(self.V[0, particle], self.V[1, particle], self.dt, self.m)

            total_forces[0, particle] = total_forces[0, particle] + forceVx
            total_forces[1, particle] = total_forces[1, particle] + forceVy

        self.acc = np.zeros(shape=(2, self.N_particles))
        self.acc[0, :] = total_forces[0, :] / self.m
        self.acc[1, :] = total_forces[1, :] / self.m


        return np.array([self.pos[0, particleID], self.pos[1, particleID], self.V[0, particleID], self.V[1, particleID]])

    def step(self, action):

        reward = self.rewardFailed
        done = False
        offboundary = False
        info = False
        selfDrivenForce = self.selfDrivenForce


        particleID = 0
        for particle in self.fluid_particles_id:

            self.pos[0, particle] = self.pos[0, particle] + self.dt * self.V[0, particle] + ((self.dt ** 2) * 0.5) * self.acc[0, particle]
            self.pos[1, particle] = self.pos[1, particle] + self.dt * self.V[1, particle] + ((self.dt ** 2) * 0.5) * self.acc[1, particle]
            self.moveParticleCell(particle)

            V_half = np.zeros(shape=(2, self.N_particles))

            V_half[0, particle] = self.V[0, particle] + (0.5 * self.dt) * self.acc[0, particle]
            V_half[1, particle] = self.V[1, particle] + (0.5 * self.dt) * self.acc[1, particle]

            dist_up = LA.norm(self.pos[:, particle] - self.pos_up_wall)
            dist_bottom = LA.norm(self.pos[:, particle] - self.pos_bottom_wall)

            entrance = (dist_up < (self.radiuswall_up - self.entranceTrajectoryPenalty)) or (dist_bottom < (self.radiuswall_bottom - self.entranceTrajectoryPenalty))

            particleID = particle
            if(self.pos[0,particle] < self.WALL_LEFT) and entrance:
                reward = self.rewardSuccess
                done = True

            elif(self.pos[0,particle] > self.WALL_RIGHT) and entrance:
                reward = self.rewardSuccess
                done = True

            if(self.pos[1,particle] < self.WALL_BOTTOM) and entrance:
                reward = self.rewardSuccess
                done = True

            elif(self.pos[1,particle] > self.WALL_UP) and entrance:
                reward = self.rewardSuccess
                done = True

            if ((self.pos[1,particle] > self.L) or (self.pos[1,particle] < 0) or (self.pos[0,particle] > self.L) or (self.pos[0,particle] < 0)) and not done:
                reward = self.rewardSuccess
                offboundary = True

            if (not offboundary) and (not done):
                    #self.computeCellParticlesAll()

                    total_forces = np.zeros(shape=(2, self.N_particles))

                    eps = np.random.normal(0, 1, 1)

                    for particle in self.fluid_particles_id:

                        rij, entrance = self.computeRandEntrance(particle)

                        if (rij < self.rc) and not (entrance):
                            cellId = self.computeCell(self.pos[:, particle])

                            for q in range(9):
                                neighbour_cell = self.neighbours[cellId, q]

                                neighbour_particles_all = np.where(self.cell_particles == neighbour_cell)
                                for p in range(len(neighbour_particles_all[0])):
                                    neighbour_particles_single = neighbour_particles_all[0][p]

                                    rij_vec_x = self.pos[0, particle] - self.pos[0, neighbour_particles_single]
                                    rij_vec_y = self.pos[1, particle] - self.pos[1, neighbour_particles_single]

                                    rij = np.sqrt(rij_vec_x ** 2 + rij_vec_y ** 2)

                                    if (rij < self.rc) and (neighbour_particles_single != particle):
                                        type_neighbour = self.TYPE_WALL
                                        rij_hat_x = rij_vec_x / rij
                                        rij_hat_y = rij_vec_y / rij

                                        tij_vec_x = -rij_vec_y
                                        tij_vec_y = rij_vec_x
                                        tij_hat_x = - rij_hat_y
                                        tij_hat_y = rij_hat_x

                                        dij = (self.di + self.di) / 2

                                        [forceAx, forceAy] = self.avoidance_coordinates(rij, dij, rij_hat_x, rij_hat_y)

                                        vij_x = self.V[0, particle] - 0
                                        vij_y = self.V[1, particle] - 0

                                        #not used
                                        uij_hat_x = vij_x * tij_hat_x
                                        uij_hat_y = vij_y * tij_hat_y

                                        [forceCx, forceCy] = self.compression_coordinates(rij, dij, rij_hat_x,
                                                                                          rij_hat_y)

                                        [forceFx, forceFy] = self.friction_coordinates(rij, dij, tij_hat_x, tij_hat_y,
                                                                                       tij_vec_x,
                                                                                       tij_vec_y, vij_x, vij_y)

                                        total_forces[0, particle] = total_forces[
                                                                        0, particle] + forceAx + forceCx + forceFx
                                        total_forces[1, particle] = total_forces[
                                                                        1, particle] + forceAy + forceCy + forceFy


                    [forceVx, forceVy] = self.viscous_force(self.V[0, particle], self.V[1, particle], self.dt, self.m)

                    if(self.continuous):

                        #normalization
                        action = ((action+1)/2)*np.pi

                        #clip angle between 0 and 2pi)
                        upperBound = np.pi
                        action = np.clip(action, 0, upperBound)

                        forceSx = (selfDrivenForce * np.cos(action) *self.m)/self.dt
                        forceSy = (selfDrivenForce * np.sin(action) *self.m)/self.dt
                    else:

                        #project the forces in X and Y
                        forceSx = (selfDrivenForce * self.coss[action] * self.m) / self.dt
                        forceSy = (selfDrivenForce * self.sins[action] * self.m) / self.dt

                    total_forces[0, particle] = total_forces[0, particle]+ forceSx + forceVx
                    total_forces[1, particle] = total_forces[1, particle]+ forceSy + forceVy

                    acc_2 = np.zeros(shape=(2,self.N_particles))
                    acc_2[0,:] = total_forces[0,:] / self.m
                    acc_2[1,:] = total_forces[1,:] / self.m

                    self.V[0,:] = self.V[0,:] + (self.dt*0.5)*(self.acc[0,:] + acc_2[0,:])
                    self.V[1,:] = self.V[1,:] + (self.dt*0.5) * (self.acc[1, :] + acc_2[1, :])

                    self.resetWallVelocities()
                    self.acc = acc_2

            if (offboundary):
                if(not done):
                    print('OFFBOUNDARY')
                    print('x', self.pos[0, particleID])
                    print('y', self.pos[1, particleID])
                    self.pos[0, particleID], self.pos[1, particleID], self.V[0, particleID], self.V[
                        1, particleID] = 0, 0, 0, 0;
                    reward = -10
                    done = False
                    info = True

        return np.array([self.pos[0, particleID], self.pos[1, particleID], self.V[0, particleID], self.V[1, particleID]]), reward, done, info




