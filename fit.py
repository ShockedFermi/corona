"""Corona. Spread of the coronavirus.

Usage:
    corona.py <country> [--deaths --log]

Options:
    -l --log        make log plot
    -d --deaths     plot deaths
"""

from docopt import docopt
if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)

import math
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as pl
data = __import__(args['<country>'])

def propagate_sigmas( Nis ):
    s = np.zeros( len(Nis) )
    s[0] = np.sqrt(Nis[0])
    for i, y in enumerate(Nis):
        if i == 0: continue
        delta_y = y - Nis[i - 1]
        s[i] = np.sqrt( delta_y + s[i - 1]*s[i - 1])
    return s
        
#Prepare data

#Ni = number of infected
Nis       = np.array(list(data.DATA.values()))[:, 1*args['--deaths'] ]
Ni_sigmas = propagate_sigmas( Nis )
days      = np.arange(len(Nis)) - len(Nis) + 1

print("{0} days of data: {1}".format( len(days), days ) )
print("            data: {0}".format( Nis ) )
print("ni sigmas:", Ni_sigmas )

#models for the total number of infections

def logistic( t, L, k, t_0 ):
    """
    L is the final number of infections: [current number, POPULATION]
    k is the logistic growth rate: [current_rate*4/pop, inf]
    x_0 is the time of inflection: [current day, inf)
    """
    #print("t {0} L {1} k {2} t_0 {3}".format( t, L, k, t_0))
    return L/(1 + np.exp(-(t - t_0)/k ) )

#these are now identical...

def model( t, pop, K, t_inflection):
    """
    Pop is the final number of infections: [current_number, POPULATION]
    K is the contamination factor: [current_rate*4/pop, inf]
    t_infletion is the time of inflection point
    """
    return pop/(1 + np.exp((t_inflection - t)*K))

def model_s(t, pop, K, t_inflection, pop_s, K_s, t_inflection_s ):
    return model(t, pop, K, t_inflection)*np.sqrt(pop_s*pop_s/pop/pop + t_inflection_s*t_inflection_s*K*K + K_s*K_s*(t - t_inflection)**2)

def exponential( t, tau, A ):
    return A*np.exp(t/tau)

def P_infection( N, p_infected, p_contamination):
    return 100*(1 - (1 - p_infected*p_contamination)**N)

def P_flat_infection( N_inf_now, N_inf_then, pop):
    return 100*(N_inf_then - N_inf_now) / (pop - N_inf_now)










def fit_data( first_day = -1e3, last_day = 0):

    first_i = len(Nis) + first_day if -first_day < len(Nis) else 0
    last_i  = len(Nis) + last_day - 1
    
    current_infected = Nis[last_i]
    current_rate = current_infected - Nis[last_i - 1]
    current_growth = current_rate * 4 / data.POPULATION

    #remove early data:
    cut_factor = 10.
    icut = 0
    while Nis[icut] < Nis[-1]/cut_factor:
        icut += 1
    first_i += icut

    """
    fit_log_v, fit_log_cov = curve_fit(logistic,
                                       days[first_i:last_i + 1],
                                       Nis[first_i:last_i + 1],
                                       #p0    = [current_infected, current_growth, 7],
                                       sigma = Ni_sigmas[first_i:last_i + 1],
                                       bounds=([current_infected, current_growth, 0],
                                               [np.inf,           np.inf,         np.inf]),
                                       maxfev = 1e6,
                                       verbose = True)
    print("Fitted values logistic: {}".format(fit_log_v) )
    fit_log_s = np.sqrt( np.diag( fit_log_cov ) )
    """

    print("Fitting from {0} days ago until {1} days ago".format( first_i - len(Nis), last_i + 1 - len(Nis)) )
          
    fit_mod_v, fit_mod_cov = curve_fit(model,
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
    print("Fitted values model: {}".format(fit_mod_v) )
    fit_mod_s = np.sqrt( np.diag( fit_mod_cov ) )

    fit_exp_v, fit_exp_cov = curve_fit(exponential,
                                       days[first_i:last_i + 1],
                                       Nis[first_i:last_i + 1],
                                       sigma = Ni_sigmas[first_i:last_i + 1],
                                       bounds=(0, np.inf) )
    print("Fitted values exponential: {}".format(fit_exp_v) )
    fit_exp_s = np.sqrt( np.diag( fit_exp_cov ) )
    

    fit_x = np.concatenate( (days, np.arange( 30 ) ) )

    #data
    pl.errorbar( days, Nis, yerr = Ni_sigmas, color='b' )

    #model fit
    fit_mod_y   = model( fit_x, *fit_mod_v )
    fit_mod_err = model_s( fit_x, *fit_mod_v, *fit_mod_s)
    print( 'fit_mod_err', fit_mod_err )
    pl.errorbar( fit_x,
                 fit_mod_y,
                 yerr = fit_mod_err,
                 linewidth=0.3,
                 color='k',
                 linestyle='--')
    pl.fill_between( fit_x,
                     fit_mod_y - fit_mod_err,
                     fit_mod_y + fit_mod_err)
    #pl.plot( fit_x, model( fit_x, fit_mod_v[0] + fit_mod_s[0], fit_mod_v[1] + fit_mod_s[1], fit_mod_v[2] + fit_mod_s[2] ), linewidth=0.3, color='r')
    #pl.plot( fit_x, model( fit_x, fit_mod_v[0] - fit_mod_s[0], fit_mod_v[1] - fit_mod_s[1], fit_mod_v[2] - fit_mod_s[2] ), linewidth=0.3, color='g')

    
    #exponential fit
    pl.plot( fit_x[:-25], exponential( fit_x[:-25], *fit_exp_v ), linewidth=0.3, color='m')
    
    pl.xticks( rotation='vertical' )
    pl.xlabel("Days from today")
    pl.ylabel("Number of " + ["infections", "deaths"][args['--deaths']] + " in " + args['<country>'])
    if args['--log']:
        pl.yscale('log')
    else:
        pl.ylim( 0, Nis[-1] * 4 )
    pl.grid(True, 'both')

    print( "Model predicts a total of {0:.0f} +/- {1:.1f} {2}".format( fit_mod_v[0], fit_mod_s[0], ["infections", "deaths"][args['--deaths']]) )
    print( "a contamination factor (P_infection * N_contacts) of {0:.2f} +/- {1:.1f}".format( fit_mod_v[1], fit_mod_s[1]) )
    print( "an inflection point in {0:.2f} +/- {1:.1f} days".format( fit_mod_v[2], fit_mod_s[2]) )

    print( "Chance of {1} in next two weeks (best case scenario) {0:.2}%".format(P_flat_infection( list(data.DATA.values())[-1][1*args['--deaths']], model( 14, *fit_mod_v ), data.POPULATION ), ["infection", "death"][args['--deaths']] ) )
    print( "Chance of {1} in next two weeks (worst case scenario) {0:.2}%".format(P_flat_infection( list(data.DATA.values())[-1][1*args['--deaths']], exponential( 14, *fit_exp_v ), data.POPULATION ), ["infection", "death"][args['--deaths']] ) )

    pl.show( )

fit_data(-1e3, 0)
