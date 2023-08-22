from re import S
import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

from robotics.robot import Robot

def build_mass_matrix(I1, I2, m2, l1, l2, c2):
    M = np.array([
        [I1 + I2 + m2*l1**2 + 2*m2*l1*l2/2*c2, I2 + m2*l1*l2/2*c2],
        [I2 + m2*l1*l2/2*c2, I2]
    ])
    return M
def build_centrifugal_matrix(m2, l1, l2, s2, d_phi1, d_phi2):
    C = np.array([
        [-2*m2*l1*l2/2*s2*d_phi2, -m2*l1*l2/2*s2*d_phi2],
        [m2*l1*l2/2*s2*d_phi1, 0]
    ])
    return C
def build_gravity_vector(g, m1, m2, l1, l2, s1, s12):
    tau = np.array([
        -m1*g*l1/2*s1 - m2*g*(l1*s1 + l2/2*s12),
        -m2*g*l2/2*s12
    ])
    return tau

class Arm(Robot):

    d, m = 4, 1

    g = 9.8

    K = np.array([[.5, .5]])


    # @staticmethod
    # def observe(x):
    #     phi1, d_phi1, phi2, d_phi2 = x
    #     cphi1, sphi1 = np.cos(phi1), np.sin(phi1)
    #     cphi2, sphi2 = np.cos(phi2), np.sin(phi2)
    #     obs = np.array([cphi1, sphi1, d_phi1, cphi2, sphi2, d_phi2])
    #     return obs

    def __init__(self, I2, m2, alpha, dt=0.02, sigma=0):
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0])
        super().__init__(self.x0, self.d, self.m, dt, sigma=sigma)

        self.m2 = m2
        self.I2 = I2
        self.m1 = 1.0
        self.l1 = 1.0
        self.l2 = 1.0
        self.I1 = self.m1*self.l1**2/3
        self.I = self.I1 + self.I2
        self.alpha = alpha
        # self.l1 = 1
        # self.m1 = 1
        # self.m2 = 1
        # self.l2 = 1
        # self.I1 = self.m1*self.l1**2/3
        # self.I2 = self.m2*self.l2**2/3
        self.I = self.I1 + self.I2
        # self.alpha = .2

        # print(f'm1 = {m1}, m2 = {m2}, l1 = {l1}, l2 = {l2}')

        self.omega_2 = self.g/self.l1
        self.period = 2*np.pi * np.sqrt(1 / self.omega_2)


        self.phi_max = np.inf
        self.dphi_max = 10.0
        self.x_min = np.array([-self.phi_max, -self.dphi_max, -self.phi_max, -self.dphi_max])
        self.x_max = np.array([+self.phi_max, +self.dphi_max, +self.phi_max, +self.dphi_max])

    def rigid_body(self, cphi1, sphi1, cphi2, sphi2, d_phi1, d_phi2):
        s12 = cphi1*sphi2 + cphi2*sphi1
        M = build_mass_matrix(self.I1, self.I2, self.m2, self.l1, self.l2, cphi2)
        C = build_centrifugal_matrix(self.m2, self.l1, self.l2, sphi2, d_phi1, d_phi2)
        tau = build_gravity_vector(self.g, self.m1, self.m2, self.l1, self.l2, sphi1, s12)
        return M, C, tau

    def forward_dynamics(self, x, u):
        phi1, d_phi1, phi2, d_phi2 = x
        # print(x)
        cphi1, sphi1 = np.cos(phi1), np.sin(phi1)
        cphi2, sphi2 = np.cos(phi2), np.sin(phi2)
        M, C, tau = self.rigid_body(cphi1, sphi1, cphi2, sphi2, d_phi1, d_phi2)
        # print(f'M = {M}')
        # print(f'C = {C}')
        # print(f'tau = {tau}')
        d_phi = np.array([d_phi1, d_phi2])
        # Bu = np.array([0, u.squeeze()]) - self.alpha * np.array([d_phi1**2, 0])
        Bu = np.array([0, u.squeeze()]) - self.alpha * d_phi

        dd_phi1, dd_phi2 = np.linalg.solve(M, tau + Bu - C@d_phi)
        # print(f'dd_phi1 {dd_phi1}')
        # print(f'dd_phi2 {dd_phi2}')

        return np.array([d_phi1, dd_phi1, d_phi2, dd_phi2])
    
    def inverse_dynamics(self, x, x_dot):
        phi1, d_phi1, phi2, d_phi2 = x
        d_phi1, dd_phi1, d_phi2, dd_phi2 = x_dot
        cphi1, sphi1 = np.cos(phi1), np.sin(phi1)
        cphi2, sphi2 = np.cos(phi2), np.sin(phi2)
        d_phi = np.array([d_phi1, d_phi2])
        dd_phi = np.array([dd_phi1, dd_phi2])

        M, C, tau = self.rigid_body(cphi1, sphi1, cphi2, sphi2, d_phi1, d_phi2)

        Bu = M@dd_phi +C@d_phi -tau

        return Bu
    
    def compute_tip_positions(self, state_values):
        phi1, d_phi1, phi2, d_phi2 = state_values.T
        cphi1, sphi1 = np.cos(phi1), np.sin(phi1)
        cphi2, sphi2 = np.cos(phi2), np.sin(phi2)
        c12 = cphi1*cphi2 - sphi1*sphi2
        s12 = cphi1*sphi2 + cphi2*sphi1
        abscissa = self.l1*sphi1 + self.l2*s12
        ordinate = -self.l1*cphi1-self.l2*c12
        return np.stack((abscissa, ordinate), axis=1)
    
    def evaluate_tracking(self, x_values, x_target_values):
    
        tip_values = self.compute_tip_positions(x_values)
        tip_target_values = self.compute_tip_positions(x_target_values)
        e_values = tip_values - tip_target_values
        error_values = np.linalg.norm(e_values, axis=1)

        return error_values
    


    def plot_system(self, x, u, t, **kwargs):
        phi1, d_phi1, phi2, d_phi2 = x[0], x[1], x[2], x[3]
        # push = 0.7*np.sign(np.mean(u))
        cphi1, sphi1 = np.cos(phi1), np.sin(phi1)
        cphi2, sphi2 = np.cos(phi2), np.sin(phi2)
        c12 = cphi1*cphi2 - sphi1*sphi2
        s12 = cphi1*sphi2 + cphi2*sphi1

        plt.arrow(
            self.l1*sphi1 + self.l2*s12,
            -self.l1*cphi1 - self.l2*c12,
            0.05*u[0]*c12,
            0.05*u[0]*s12,
            color='red',
            head_width=0.1,
            head_length=0.01*abs(u[0]),
            alpha=0.5)

        plt.plot([0, self.l1*sphi1], [0, -self.l1*cphi1], color='black', **kwargs)
        plt.plot([self.l1*sphi1, self.l1*sphi1 + self.l2*s12],
                [-self.l1*cphi1, -self.l1*cphi1-self.l2*c12], color='blue', **kwargs)
        total_length=self.l1+self.l2
        plt.xlim((-(1.5*total_length), 1.5*total_length))
        plt.ylim((-(1.5*total_length), 1.5*total_length))
        plt.gca().set_aspect('equal', adjustable='box')