"""Fit the spread of the coronavirus.

Usage:
    analyse.py <country>             [--deaths --log --y0=<n> --y1=<n>]

Options:
    --log                   make log plot
    --deaths                plot deaths
    --y0=<n>                fit to data starting at x > n [default: 10]
    --y1=<n>                fit to data ending at x > n [default: 10000000000]
"""

from docopt import docopt
if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)

import math
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as pl

from coronalib import *
data = __import__(args['<country>'])
        
#Prepare data
#Ni = number of infected
Nis       = np.array(list(data.DATA.values()))[:, 1*args['--deaths'] ]
Ni_sigmas = propagate_sigmas( Nis )
days      = np.arange(len(Nis)) - len(Nis) + 1

print("{0} days of data: {1}".format( len(days), days ) )
print("            data: {0}".format( Nis ) )
print("ni sigmas:", Ni_sigmas )



#select data range:
first_i = 0
while Nis[first_i] < int(args['--y0']):
    first_i += 1

last_i = 0
while last_i + 1 != len(Nis) and Nis[last_i] < int(args['--y1']):
    last_i += 1

current_infected = Nis[last_i]
current_rate     = current_infected - Nis[last_i - 1]
current_growth   = current_rate * 4 / data.POPULATION
    
print("Fitting from {0} days ago until {1} days ago".format( first_i - len(Nis), last_i + 1 - len(Nis)) )
    
fit_mod_v, fit_mod_cov = curve_fit(logistic,
                                   days[first_i:last_i + 1],
                                   Nis[first_i:last_i + 1],
                                   p0    = [current_infected, 0.5, 7],
                                   sigma = Ni_sigmas[first_i:last_i + 1],
                                   bounds=([current_infected,   0.0, -len(Nis)],
                                           [data.POPULATION, np.inf,   np.inf]),
                                   maxfev = 1e6,
                                   ftol = 1e-15,
                                   xtol = 1e-15,
                                   verbose = True)
print("Fitted [Ni_tot, K, t_inflection] logistic: {}".format(fit_mod_v) )
fit_mod_s = np.sqrt( np.diag( fit_mod_cov ) )


fit_exp_v, fit_exp_cov = curve_fit(exponential,
                                   days[first_i:last_i + 1],
                                   Nis[first_i:last_i + 1],
                                   sigma = Ni_sigmas[first_i:last_i + 1],
                                   bounds=(0, np.inf),
                                   maxfev = 1e6)
print("Fitted [K, N_0] exponential: {}".format(fit_exp_v) )
fit_exp_s = np.sqrt( np.diag( fit_exp_cov ) )



#  - plotting -



fit_x = np.concatenate( (days, np.arange( 30 ) ) )

#logistic fit
fit_mod_y   = logistic( fit_x, *fit_mod_v )
fit_mod_err = logistic_s( fit_x, *fit_mod_v, *fit_mod_s)

logistic_pl, = pl.plot( fit_x,
                     fit_mod_y,
                     linewidth=1.0,
                     color='c',
                     linestyle='--')
pl.fill_between( fit_x,
                 fit_mod_y - fit_mod_err,
                 fit_mod_y + fit_mod_err)

#exponential fit
exp_pl, = pl.plot( fit_x[:-25], exponential( fit_x[:-25], *fit_exp_v ), linewidth=0.5, color='m')

#data
data_pl = pl.errorbar( days, Nis, yerr = Ni_sigmas, color='k' )[0]

exp_3, = pl.plot( fit_x[:-25], exponential( fit_x[:-25], 1/3., Nis[last_i] ), linewidth = 0.3, color='r' )
exp_5, = pl.plot( fit_x[:-25], exponential( fit_x[:-25], 1/5., Nis[last_i] ), linewidth = 0.3, color='g' )

pl.title("Corona in {0}, {1}".format( args['<country>'], list(data.DATA.keys())[-1]) ) 
pl.xticks( rotation='vertical' )
pl.xlabel("Days from today")
pl.ylabel("Number of " + ["infections", "deaths"][args['--deaths']] + " in " + args['<country>'])
if args['--log']:
    pl.yscale('log')
else:
    pl.ylim( 0, Nis[-1] * 4 )

pl.grid(True, 'both')
    
data_pl.set_label( "Data" )
logistic_pl.set_label( "Logistic curve fit" )
exp_pl.set_label( "Exponential fit" )
exp_3.set_label( "Exponential K = 1/3" )
exp_5.set_label( "Exponential K = 1/5" )
    
pl.legend()

print( " -- MODEL PREDICTIONS --")
print( "\t Total {2}: {0:.0f} +/- {1:.1f}".format( fit_mod_v[0], fit_mod_s[0], ["infections", "deaths"][args['--deaths']]) )
print( "\t Contamination factor: {0:.3f} +/- {1:.2f} infections per infected per day".format( fit_mod_v[1], fit_mod_s[1]) )
print( "\t Inflection point {0:.2f} +/- {1:.1f} days".format( fit_mod_v[2], fit_mod_s[2]) )

print( "Chance of {1} in next two weeks (best case scenario) {0:.2}%".format(P_flat_infection( list(data.DATA.values())[-1][1*args['--deaths']], logistic( 14, *fit_mod_v ), data.POPULATION ), ["infection", "death"][args['--deaths']] ) )
print( "Chance of {1} in next two weeks (worst case scenario) {0:.2}%".format(P_flat_infection( list(data.DATA.values())[-1][1*args['--deaths']], exponential( 14, *fit_exp_v ), data.POPULATION ), ["infection", "death"][args['--deaths']] ) )

pl.show()
