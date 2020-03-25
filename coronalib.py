import numpy as np

def propagate_sigmas( Nis ):
    s = np.zeros( len(Nis) )
    s[0] = np.sqrt(Nis[0])
    for i, y in enumerate(Nis):
        if i == 0: continue
        delta_y = y - Nis[i - 1]
        s[i] = np.sqrt( delta_y + s[i - 1]*s[i - 1])
    return s


#models for the total number of infections

"""
def exp_log( t, Ni_tot, K0, K1, t_inflection, t_measures):
   
    Pop is the final number of infections: [current_number, POPULATION]
    K is the contamination factor: [current_rate*4/pop, inf]
    t_infletion is the time of inflection point
    t_measures is the time at which K changes
   
    if( t < t_measures):
        return exponential( t, K0, 
    else:
        return logistic( t, Ni_tot, K1, t_inflection )
"""
    
def logistic( t, Ni_tot, K, t_inflection):
    """
    Ni_tot is the final number of infections: [current_number, POPULATION]
    K is the contamination factor: [current_rate*4/pop, inf]
    t_infletion is the time of inflection point
    """
    return Ni_tot/(1 + np.exp((t_inflection - t)*K))

def logistic_s(t, pop, K, t_inflection, pop_s, K_s, t_inflection_s ):
    return logistic(t, pop, K, t_inflection)*np.sqrt(pop_s*pop_s/pop/pop + t_inflection_s*t_inflection_s*K*K + K_s*K_s*(t - t_inflection)**2)

def exponential( t, K, N_0 ):
    """
    K is the contamination factor
    N_0 is the number of infected at t = 0
    """
    return N_0*np.exp(t*K)

def linear( t, a, b ):
    return a + b*t

def P_infection( N, p_infected, p_contamination):
    return 100*(1 - (1 - p_infected*p_contamination)**N)

def P_flat_infection( N_inf_now, N_inf_then, pop):
    return 100*(N_inf_then - N_inf_now) / (pop - N_inf_now)
