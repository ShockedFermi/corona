"""Fit the spread of the coronavirus.

Usage:
    analyse.py <country>... [--deaths --log --y0=<n> --y1=<n>]

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
import matplotlib
import matplotlib.pyplot as pl

from coronalib import *

#Prepare data
#Ni = number of infected
today = ""

population = []
Nis        = []
Ni_sigmas  = []
days       = []
ifirst     = []
ilast      = []
cmap       = []
for i, country in enumerate( args['<country>']):
    print("Importing data from {0}".format( country ) ) 
    data = __import__( country )
    if i == 0: today = list(data.DATA.keys())[-1]
    population.append(data.POPULATION)
    cmap.append(matplotlib.cm.get_cmap(data.CMAP))
    Nis.append(       np.array(list(data.DATA.values()))[:, 1*args['--deaths'] ] )
    Ni_sigmas.append( propagate_sigmas( Nis[i] ) )
    days.append(      np.arange(len(Nis[i])) - len(Nis[i]) + 1 )

    print("{0} days of data: {1}".format( len(days[i]), days[i] ) )
    print("            data: {0}".format( Nis[i]) )
    print("ni sigmas:", Ni_sigmas[i] )



    #select data range:
    ifirst.append(0)
    while Nis[i][ifirst[i]] < int(args['--y0']):
        ifirst[i] += 1

    ilast.append(0)
    while ilast[i] + 1 != len(Nis[i]) and Nis[i][ilast[i]] < int(args['--y1']):
        ilast[i] += 1
    if i != 0:
        print( "days before:", days[i] )
        days[i] += days[0][ilast[0]] - days[i][ilast[i]]
        print( "days after:", days[i], " ilast[i]: ", ilast[i] )

    print("Fitting from {0} days ago until {1} days ago".format( ifirst[i] - len(Nis[i]), ilast[i] + 1 - len(Nis[i])) )

    fit_piece_v, fit_piece_cov = curve_fit(logistic_piece,
                                         days[i][ifirst[i]:ilast[i] + 1],
                                         Nis [i][ifirst[i]:ilast[i] + 1],
                                         p0    = [ 0,
                                                   7,
                                                   0.5,
                                                   0.5,
                                                   population[i],
                                                   Nis[i][ilast[i]] ],
                                         sigma = Ni_sigmas[i][ifirst[i]:ilast[i] + 1],
                                         bounds=([-len(Nis[i]),
                                                  -len(Nis[i]),
                                                  0.0,
                                                  0.0,
                                                  Nis[i][ilast[i]],
                                                  0],
                                                 [ np.inf,
                                                   np.inf,
                                                   np.inf,
                                                   np.inf,
                                                   population[i],
                                                   population[i]]),
                                         maxfev = 1e6,
                                         ftol   = 1e-15,
                                         xtol   = 1e-15,
                                         verbose = True)
    print("Fitted [t_measures, t_inflection, K0, K1, Ni_inf, N0] logistic: {}".format(fit_piece_v) )
    fit_piece_s = np.sqrt( np.diag( fit_piece_cov ) )
    
    fit_log_v, fit_log_cov = curve_fit(logistic,
                                       days[i][ifirst[i]:ilast[i] + 1],
                                       Nis [i][ifirst[i]:ilast[i] + 1],
                                       p0    = [ 7,
                                                 0.5,
                                                 Nis[i][ilast[i]] ],
                                       sigma = Ni_sigmas[i][ifirst[i]:ilast[i] + 1],
                                       bounds=([-len(Nis[i]),
                                                0.0,
                                                Nis[i][ilast[i]] ],
                                               [np.inf,
                                                np.inf,
                                                population[i] ]),
                                       maxfev = 1e6,
                                       ftol   = 1e-15,
                                       xtol   = 1e-15,
                                       verbose = True)
    print("Fitted [t_inflection, K, Ni_tot] logistic: {}".format(fit_log_v) )
    fit_log_s = np.sqrt( np.diag( fit_log_cov ) )

    
    fit_exp_v, fit_exp_cov = curve_fit(exponential,
                                       days[i][ifirst[i]:ilast[i] + 1],
                                       Nis [i][ifirst[i]:ilast[i] + 1],
                                       sigma = Ni_sigmas[i][ifirst[i]:ilast[i] + 1],
                                       bounds=(0, np.inf),
                                       maxfev = 1e6)
    print("Fitted [K, N_0] exponential: {}".format(fit_exp_v) )
    fit_exp_s = np.sqrt( np.diag( fit_exp_cov ) )



    #  - plotting -


    fit_x = np.concatenate( (days[i], np.arange( 30 ) ) )

    #data
    data_pl = pl.errorbar( days[i],
                           Nis[i],
                           yerr = Ni_sigmas[i],
                           color=cmap[i](0.8))[0]
    data_pl.set_label( "Data {0}{1}".format(args['<country>'][i], ["", " (+" + str(days[i][-1]) + " days)"][i!=0] ) )
    
    #logistic fit
    fit_log_y   = logistic(   fit_x, *fit_log_v )
    fit_log_err = logistic_s( fit_x, *fit_log_v, *fit_log_s)
    
    logistic_pl, = pl.plot( fit_x,
                            fit_log_y,
                            linewidth=1.0,
                            color=cmap[i](0.5))
    pl.fill_between( fit_x,
                     fit_log_y - fit_log_err,
                     fit_log_y + fit_log_err,
                     color = cmap[i](0.1))
    logistic_pl.set_label( "Logistic fit {0}".format(args['<country>'][i]) )

    #exponential fit
    exp_pl, = pl.plot( fit_x[:-14],
                       exponential( fit_x[:-14], *fit_exp_v ),
                       linewidth=1.0,
                       color=cmap[i](0.25),
                       linestyle = '--')
    exp_pl.set_label( "Exponential fit {0}".format(args['<country>'][i]) )

    if i == 0:
        exp_3, = pl.plot( fit_x[:-14], exponential( fit_x[:-14], 1/3., Nis[i][ilast[i]] ), linewidth = 1.0, linestyle = ':', color='r' )
        exp_5, = pl.plot( fit_x[:-14], exponential( fit_x[:-14], 1/5., Nis[i][ilast[i]] ), linewidth = 1.0, linestyle = ':', color='g' )
        exp_3.set_label( "Exponential K = 1/3" )
        exp_5.set_label( "Exponential K = 1/5" )
    
pl.title("COVID-19 in {0}, {1}".format( ", ".join(args['<country>']), today) ) 
pl.xticks( rotation='vertical' )
pl.xlabel("Days from today")
pl.ylabel("Number of " + ["infections", "deaths"][args['--deaths']] + " in " + args['<country>'][0])

if args['--log']:
    pl.yscale('log')
else:
    pl.ylim( 0, Nis[0][-1] * 4 )

pl.grid(True, 'both')
    
pl.legend()

print( " -- MODEL PREDICTIONS FOR {0}--".format(args['<country>'][0]))
print( "\t Total {2}: {0:.0f} +/- {1:.1f}".format( fit_log_v[0], fit_log_s[0], ["infections", "deaths"][args['--deaths']]) )
print( "\t Contamination factor: {0:.3f} +/- {1:.2f} infections per infected per day".format( fit_log_v[1], fit_log_s[1]) )
print( "\t Inflection point {0:.2f} +/- {1:.1f} days".format( fit_log_v[2], fit_log_s[2]) )

print( "Chance of {1} in next two weeks (best case scenario) {0:.2}%".format(P_flat_infection( list(data.DATA.values())[-1][1*args['--deaths']], logistic( 14, *fit_log_v ), population[0] ), ["infection", "death"][args['--deaths']] ) )
print( "Chance of {1} in next two weeks (worst case scenario) {0:.2}%".format(P_flat_infection( list(data.DATA.values())[-1][1*args['--deaths']], exponential( 14, *fit_exp_v ), population[0] ), ["infection", "death"][args['--deaths']] ) )

pl.show()
