from re import S
import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

from robotics.robot import Robot




class Cartpole(Robot):

    d, m = 4, 1


    K = np.array([[1., .1, .2, .2, .1]])


    @staticmethod
    def observe(x):
        y, d_y, phi, d_phi = x
        cphi, sphi = np.cos(phi), np.sin(phi)
        obs = np.array([y, d_y, cphi, sphi, d_phi])
        return obs

    @staticmethod
    def get_state(obs):
        y, d_y, cphi, sphi, d_phi = torch.unbind(obs, dim=1)
        phi = torch.atan2(sphi, cphi)
        x = torch.stack((y, d_y, phi, d_phi), dim=1)
        return x

    def __init__(self, mass, Mass, l, alpha, beta, dt=0.02, g=9.8):
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0])
        super().__init__(self.x0, self.d, self.m, dt)

        self.g = g
        self.l = l
        self.mass = mass
        self.Mass = Mass
        self.total_mass = Mass + mass
        self.alpha = alpha
        self.beta = beta

        self.omega_2 = g/l
        self.period = 2*np.pi * np.sqrt(l / g)


        self.dy_max = 10.0
        self.dphi_max = 10.0
        self.x_min = np.array([-np.inf, -self.dy_max, -np.inf, -self.dphi_max])
        self.x_max = np.array([+np.inf, +self.dy_max, +np.inf, +self.dphi_max])

        self.R = 0.001

    def forward_dynamics(self, x, u):
        # print(x)
        y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
        force = u[0]
        cphi, sphi = np.cos(phi), np.sin(phi)

        friction_phi = - self.alpha * d_phi
        friction_y = - self.beta * self.total_mass * self.g * np.tanh(d_y)

        dd_y = self.mass*sphi*(self.l*d_phi**2 +self.g*cphi)
        dd_y += friction_y + friction_phi*cphi/self.l
        dd_y += force
        dd_y /=  self.Mass + self.mass*sphi**2

        dd_phi =  -self.mass*self.l*d_phi**2*cphi*sphi - self.total_mass*self.g*sphi
        dd_phi += self.Mass*friction_phi/(self.mass*self.l) - friction_y
        dd_phi += -force*cphi
        dd_phi /= self.l*(self.Mass + self.mass* sphi**2) 

        return np.array([d_y, dd_y, d_phi, dd_phi])
    
    def inverse_dynamics(self, x, x_dot):
        y, d_y, phi, d_phi = x
        d_y, dd_y, d_phi, dd_phi = x_dot
        cphi, sphi = np.cos(phi), np.sin(phi)
        u_y =  self.total_mass * dd_y
        u_phi = self.mass*self.l*(dd_phi*cphi - sphi*d_phi**2)
        u = np.array([u_y + u_phi])
        return u
    # def inverse_dynamics(self, xdx):
    #     dd_y, cphi, sphi, d_phi, dd_phi = xdx
    #     # y, d_y, phi, d_phi = x
    #     # d_y, dd_y, d_phi, dd_phi = x_dot
    #     # cphi, sphi = np.cos(phi), np.sin(phi)
    #     u_y = (self.mass+self.Mass) * dd_y
    #     u_phi = self.mass*self.l*(dd_phi*cphi - d_phi**2*sphi)
    #     return np.array([u_y + u_phi])
    
    def derive(self, x, x_dot):
        y, d_y, phi, d_phi = x
        d_y, dd_y, d_phi, dd_phi = x_dot
        cphi, sphi = np.cos(phi), np.sin(phi)
        return np.array([dd_y, cphi, sphi, d_phi, dd_phi])
    
    def compute_tip_positions(self, state_values):
        y, d_y, phi, d_phi = state_values.T
        cphi, sphi = np.cos(phi), np.sin(phi)
        return np.stack((y+self.l*sphi, -self.l*cphi), axis=1)
    
    def evaluate_tracking(self, x_values, x_target_values):
    
        tip_values = self.compute_tip_positions(x_values)
        tip_target_values = self.compute_tip_positions(x_target_values)
        e_values = tip_values - tip_target_values
        error_values = np.linalg.norm(e_values, axis=1)

        return error_values
    


    def plot_system(self, x, u, t, **kwargs):
        y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
        push = self.l*0.06*u[0]
        side = np.sign(push)
        cphi, s_phi = np.cos(phi), np.sin(phi)
        if push != 0:
            plt.arrow(y+side*self.l/2, 0, push, 0, color='red', head_width=0.1,**kwargs)
        plt.plot([y-self.l/2, y+self.l/2], [0, 0], color='black', **kwargs)
        plt.plot([y, y+self.l*s_phi], [0, -self.l*cphi], color='blue', **kwargs)
        plt.xlim((y-2*self.l, y+2*self.l))
        plt.ylim((-2*self.l, 2*self.l))
        # plt.ylim((-2, 2))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f't = {t}')
