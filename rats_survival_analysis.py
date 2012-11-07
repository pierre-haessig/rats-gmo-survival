#!/usr/bin/python
# -*- coding: UTF-8 -*-
""" A draft survival analysis of rats
(follow up of Seralini's article on GMO toxicity)

Quick ref:
http://en.wikipedia.org/wiki/Survival_analysis
http://en.wikipedia.org/wiki/Life_table
http://en.wikipedia.org/wiki/Laboratory_rat#Sprague_Dawley_rat

Pierre Haessig — September 2012
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

# Initialize a Random Number Generator:
seed = 0
rng = np.random.RandomState(seed)
print('Random Number Generator initialized with seed=%s' % str(seed))

# Number of rats per groups in the experiment
n_rats = 10 # 10 in Séralini's article
print('Number of rats per group: %d' % n_rats)

# Properties of the Weibull distribution
# used to randomly sample death times:

# Median mortality time:
wb_med = 600 # [days] *wb_scale
# Weibull shape parameter:
wb_shape = 4 # 1 for exponential law
# Weibull scale parameter, determined from median and shape params:
wb_scale = wb_med/np.log(2)**(1./wb_shape)  # [days]
print('Weibull parameters:')
print(' * scale: %.1f days' % wb_scale)
print(' * shape: %.1f'      % wb_shape)
print(' -> median time : %.1f days' % wb_med)

# Sample the experimental groups:
death_time_ctrl = rng.weibull(wb_shape, size=10)*wb_scale
death_time_3gp = rng.weibull(wb_shape, size=(3,10))*wb_scale
color_3gp = [(1,0.85,0),(1,0.58,0),(1,0,0)]
lw_3gp = [2,2.5,3]

################################################################################
### Plot the experiment:
fig = plt.figure('mortality distribution')
fig.clear()
ax = fig.add_subplot(111, title='Mortality distribution from a Weibull sampling'
                                '\n(shape=%.1f, scale=%.0f days)' 
                                % (wb_shape, wb_scale),
                          xlabel='time of the experiment (days)',
                          ylabel='number of dead rats per group')

# Cumulative histogram of the Control Group:
ax.hist(death_time_ctrl, label='control group',
        bins=1000, cumulative=True, range=(0,wb_scale*2),
        histtype='step', color='k', lw=2, ls='dashed')

# Plot all the other groups:
for i in range(3):
    # Select the group:
    death_time = death_time_3gp[i,:]
    ax.hist(death_time, bins=1000, label='group %d' % ((i+1)),
            cumulative=True, range=(0,wb_scale*2), histtype='step',
            alpha=1, color=color_3gp[i], lw = lw_3gp[i], zorder=1+(2-i))
    ax.hist(death_time, bins=1000,
            cumulative=True, range=(0,wb_scale*2), histtype='stepfilled',
            alpha=0.0, color=color_3gp[i])
    ax.plot(death_time, np.zeros(n_rats), 'x', color=color_3gp[i])

################################################################################
# Confidence intervals

def cum_deaths(time, death_time):
    '''Cumulative number of deaths over `time`
    
    Deaths occur according to `death_time` which can be either:
     * a vector of length nb_deaths
     * a 2D array of shape (nb_experiments, nb_deaths)
    '''
    N_t = len(time)
    # force time to be a *row vector* (with 2D)
    time = time.reshape(1, N_t)
    if death_time.ndim == 1:
        # force death_time to be a *row vector*
        death_time = death_time.reshape(1,-1)
    # Grab the number of experiments and the number of deaths:
    N_exp, N_d = death_time.shape
    time = time.repeat(N_exp, axis=0)
    
    # Initialize the output:
    cum_death = np.zeros((N_exp,N_t))
    for n in range(N_d):
        dt = death_time[:,[n]]
        cum_death[:,time>dt] += 1
    #end for
    # flatten the output if input was flat (aka 1D)
    if death_time.ndim == 1:
        cum_death = cum_death[0,:]
    return cum_death
# end cum_deaths()


# Repeat 1000× the experiment with 10 rats:
dt = rng.weibull(wb_shape, size=(1000,10))*wb_scale
t = np.linspace(0, wb_scale*2, 1000)
cum = cum_deaths(t, dt)

## Smooth out the result: [silly!]
#cum += rng.uniform(low=-0.5, high=0.5, size=cum.shape)
#cum[cum<0] *= -1

cum_mean = cum.mean(axis=0)
#ax.plot(t, cum_mean, '-',color='grey', lw=3)
ax.plot(t, cum_mean, '--',color='white', label='mean mortality')
ax.plot(wb_med, n_rats/2., 'o', color='white')

# Compute and plot Confidence Intervals
ci_level1 = 80 # [%]
cum_low1 =  np.percentile(cum, 50-ci_level1/2, axis=0)
cum_high1 = np.percentile(cum, 50+ci_level1/2, axis=0)
ax.fill_between(t, cum_low1, cum_high1, color='grey', lw=0, alpha=0.2)
ci_level2 = 99 # [%]
cum_low2 =  np.percentile(cum, 50-ci_level2/2, axis=0)
cum_high2 = np.percentile(cum, 50+ci_level2/2, axis=0)
ax.fill_between(t, cum_low2, cum_high2, color='grey', lw=0, alpha=0.1)


# Fine tune the plot:
ax.set_xlim(0, 365*2)
ax.set_ylim(-0.05, n_rats)

ax.set_yticks([0,n_rats/2,n_rats])
ax.set_yticks(np.arange(n_rats+1), minor=True)
ax.grid(False, which='both')

ax.legend(loc='upper left', prop={'size':10})

# Light gray background
ax.patch.set_facecolor([0.75]*3)
ax.grid(color=[1]*3)

fig.canvas.draw()
fig.tight_layout

# Show !
plt.show()
