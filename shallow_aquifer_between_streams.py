#!/usr/bin/env python
# coding: utf-8

# ### Shallow Aquifer model

# This notebook runs a simple 1D model for groundwater flow and water table changes in a shallow aquifer above a horizontal impermeable unit. The model follows the Dupuit approximation, setting the hydraulic gradient equal to the water table gradient.

# In[ ]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


# The basic source code for the model is here. You shouldn't need to modify it, but take a look to a get a sense for how it works.

# In[ ]:


class WaterTableSimulator(object):
    """Simulate a 1D shallow aquifer between streams, atop an impermeable unit."""
    def __init__(self,
                 K = 100.,  # hydraulic conductivity (m/day)
                 R = 2e-3,  # recharge rate (m/day)
                 H0 = 0.0,  # initial height (m)
                 n = 50,    # number of nodes in the model
                 dx = 1.,   # spacing between nodes (m)
                 dt = 1e-4, # time step (days)
                 land_ht=2.0, # land surface height (m) (for display)
                ):

        self.hydraulic_conductivity = K
        self.recharge_rate = R
        self.init_height = H0
        self.init_time = 0
        self.nodes = n
        self.dx = dx
        self.dt = dt
        self.length = self.dx * self.nodes
        self.centers = np.arange(0.5*self.dx, self.length, self.dx)
        self.H = 0 * self.centers + self.init_height
        self.H[-1] = 0
        self.H[0] = 0.0
        self.u = np.zeros(len(self.H) - 1)
        self.q = np.zeros(len(self.H) - 1)
        self.land_ht = land_ht

        # Construct an analytic solution for the steady state water table
        # NOTE: Currently not used
        self.Hanalytic = np.sqrt(np.abs((R/K) * ((self.length + self.dx/2)**2
                                                 - self.centers**2)))
        self.Qanalytic = self.centers * R
        self.Uanalytic = np.zeros(len(self.Qanalytic))
        self.Uanalytic[:-1] = self.Qanalytic[:-1] / self.Hanalytic[:-1]

        self.current_time = 0.0

    def run_one_step(self):
        """Run one time step."""
        self.dH = np.diff(self.H)      # differences in water table height btwn adjacent nodes
        self.dHdx = self.dH / self.dx  # water table gradient (= hydraulic gradient)
        self.u = -self.hydraulic_conductivity * self.dHdx  # Darcian velocity, m/day
        Hedge = 0.5 * (self.H[:-1] + self.H[1:])
        self.q = self.u * Hedge # Unit discharge, m2/day
        self.dHdt = self.recharge_rate - np.diff(self.q) / self.dx # Rate of WT rise/fall, m/day
        self.H[1:-1] += self.dHdt * self.dt  # New WT height for this time step, m
        self.H[:] = np.minimum(self.H, self.land_ht)  # water table can't go above ground
        self.current_time += self.dt   # Update current time

    def run_n_steps(self, n):
        for _ in range(n):
            self.run_one_step()
        print('Current time = ' + str(self.current_time))


# In[ ]:


def run_model(run_duration = 800.0,  # run duration (days)
                 K = 100., # hydraulic conductivity (m/day)
                 R = 2e-3, # recharge rate (m/day)
                 H0 = 0.0, # initial height (m)
                 n = 50,   # number of nodes in the model
                 dx = 10.,  # spacing between nodes (m)
                 dt = 1e-3, # time step (days)
                 plot_interval_in_days = 20,
             ):
    """Initialize, run, and display output from model.

    Parameters
    ----------
    """

    # Instantiate and initialize a simulator
    model = WaterTableSimulator(K=K, R=R, H0=H0, n=n, dx=dx, dt=dt)

    # Calculate number of animation iterations
    save_every = int(plot_interval_in_days / model.dt)
    nsteps = int(run_duration / plot_interval_in_days)

    # Set up a blank figure with placeholder lists for data
    fig, ax = plt.subplots()
    xdata = []
    ydata = []
    obj = ax.plot([], [], color = 'k')

    # For the land
    dx = model.dx
    land_ht = model.land_ht
    xland = [0.0,
             dx,
             dx,
             dx * (n-1),
             dx * (n-1),
             dx * n
            ]
    yland = [0.0, 0.0, land_ht, land_ht, 0.0, 0.0]

    # Then, set up an initialization function
    def init():
        ax.set_ylim(0, 1.5 * model.land_ht)
        ax.set_xlim(0, model.length)
        ax.set_ylabel('Height (m)')
        ax.set_xlabel('Distance (m)')
        return(obj)

    # Next, define the update function
    def update(i):
        ax.cla()
        model.run_n_steps(save_every)
        xdata = model.centers
        ydata = model.H
        ax.set_ylim(-0.1, 1.5 * model.land_ht)
        ax.set_xlim(0, model.length)
        ax.set_ylabel('Height (m)')
        ax.set_xlabel('Distance (m)')
        ax.set_title('Time = ' + str(round(model.current_time)) + ' days')
        obj = ax.fill_between(xland, yland, color='y', label='Sandstone (unsaturated)')
        obj = ax.fill_between(xdata, ydata, color='c', label='Sandstone (saturated)')
        obj = ax.plot([0.0, dx*n], [-.05, -0.05], 'k', linewidth=10, label='Impermeable shale')
        ax.legend()
        return(obj)

    # Run the animation!
    print('Running...')
    anim = FuncAnimation(fig, update, nsteps, init_func=init, blit = True)

    # Convert the animation to HTML
    vid = HTML(anim.to_jshtml())

    print('done!')

    return vid, model
