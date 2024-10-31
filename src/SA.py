from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import copy
import datetime
import math
import pickle
import random
import signal
import sys
import time
import igraph as ig
import numpy as np
import pandas as pd



#================================================================
# SA functions
#================================================================
def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))

def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)   # get hours and remainder
    m, s = divmod(s, 60)     # split remainder into minutes and seconds
    return '%4i:%02i:%02i' % (h, m, s)

class Annealer(object):
    """Performs simulated annealing by calling functions to calculate
    energy and make moves on a state.  The temperature schedule for
    annealing may be provided manually or estimated automatically.
    """

    __metaclass__ = abc.ABCMeta

    # defaults
    Tmax = 25000.0
    Tmin = 2.5
    steps = 50000
    #steps = 1000
    updates = 100
    copy_strategy = 'deepcopy'
    user_exit = False
    save_state_on_exit = False

    # placeholders
    best_state = None
    best_energy = None
    start = None

    def __init__(self, initial_state=None, load_state=None):
        if initial_state is not None:
            self.state = self.copy_state(initial_state)
        elif load_state:
            self.load_state(load_state)
        else:
            raise ValueError('No valid values supplied for neither \
            initial_state nor load_state')

        signal.signal(signal.SIGINT, self.set_user_exit)

    def save_state(self, fname=None):
        """Saves state to pickle"""
        if not fname:
            date = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
            fname = date + "_energy_" + str(self.energy()) + ".state"
        with open(fname, "wb") as fh:
            pickle.dump(self.state, fh)

    def load_state(self, fname=None):
        """Loads state from pickle"""
        with open(fname, 'rb') as fh:
            self.state = pickle.load(fh)

    @abc.abstractmethod
    def move(self):
        """Create a state change"""
        pass

    @abc.abstractmethod
    def energy(self):
        """Calculate state's energy"""
        pass

    def set_user_exit(self, signum, frame):
        """Raises the user_exit flag, further iterations are stopped
        """
        self.user_exit = True

    def set_schedule(self, schedule):
        """Takes the output from `auto` and sets the attributes
        """
        self.Tmax = schedule['tmax']
        self.Tmin = schedule['tmin']
        self.steps = int(schedule['steps'])
        self.updates = int(schedule['updates'])

    def copy_state(self, state):
        """Returns an exact copy of the provided state
        Implemented according to self.copy_strategy, one of
        * deepcopy: use copy.deepcopy (slow but reliable)
        * slice: use list slices (faster but only works if state is list-like)
        * method: use the state's copy() method
        """
        if self.copy_strategy == 'deepcopy':
            return copy.deepcopy(state)
        elif self.copy_strategy == 'slice':
            return state[:]
        elif self.copy_strategy == 'method':
            return state.copy()
        else:
            raise RuntimeError('No implementation found for ' +
                               'the self.copy_strategy "%s"' %
                               self.copy_strategy)

    def update(self, *args, **kwargs):
        """Wrapper for internal update.
        If you override the self.update method,
        you can chose to call the self.default_update method
        from your own Annealer.
        """
        self.default_update(*args, **kwargs)

    def default_update(self, step, T, E, acceptance, improvement):
        """Default update, outputs to stderr.
        Prints the current temperature, energy, acceptance rate,
        improvement rate, elapsed time, and remaining time.
        The acceptance rate indicates the percentage of moves since the last
        update that were accepted by the Metropolis algorithm.  It includes
        moves that decreased the energy, moves that left the energy
        unchanged, and moves that increased the energy yet were reached by
        thermal excitation.
        The improvement rate indicates the percentage of moves since the
        last update that strictly decreased the energy.  At high
        temperatures it will include both moves that improved the overall
        state and moves that simply undid previously accepted moves that
        increased the energy by thermal excititation.  At low temperatures
        it will tend toward zero as the moves that can decrease the energy
        are exhausted and moves that would increase the energy are no longer
        thermally accessible."""

        elapsed = time.time() - self.start
        if step == 0:
            print('\n Temperature        Energy    Accept   Improve      Steps        Elapsed   Remaining',
                  file=sys.stderr)
            print('\r{Temp:12.5f}  {Energy:12.2f}                      {Elapsed:s}            '
                  .format(Temp=T,
                          Energy=E,
                          Elapsed=time_string(elapsed)),
                  file=sys.stderr, end="")
            sys.stderr.flush()
        else:
            remain = (self.steps - step) * (elapsed / step)
            print('\r{Temp:12.5f}  {Energy:12.2f}   {Accept:7.2%}   {Improve:7.2%}  {Steps:12.2f}  {Elapsed:s}  {Remaining:s}'
                  .format(Temp=T,
                          Energy=E,
                          Accept=acceptance,
                          Improve=improvement,
                          Steps=step,
                          Elapsed=time_string(elapsed),
                          Remaining=time_string(remain)),
                  file=sys.stderr, end="")
            sys.stderr.flush()

    def anneal(self):
        """Minimizes the energy of a system by simulated annealing.
        Parameters
        state : an initial arrangement of the system
        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        self.start = time.time()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial Elapsed
        T = self.Tmax
        E = self.energy()
        prevState = self.copy_state(self.state)
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials = accepts = improves = 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)

        # Attempt moves to new states
        Temps = []
        Energy = []
        while step < self.steps and not self.user_exit:
            step += 1
            #if step % 1000 == 0:
            #    print(self.best_energy)
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                # Restore previous state
                self.state = self.copy_state(prevState)
                E = prevEnergy
            else:
                # Accept new state and compare to best state
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            if self.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(
                        step, T, E, accepts / trials, improves / trials)
                    trials = accepts = improves = 0
            if step % 100 == 1:
                Temps.append(T)
                Energy.append(-E)
        self.state = self.copy_state(self.best_state)
        if self.save_state_on_exit:
            self.save_state()

        # Return best state and energy
        return Temps, Energy, self.best_state, self.best_energy

    def auto(self, minutes, steps=2000):
        """Explores the annealing landscape and
        estimates optimal temperature settings.
        Returns a dictionary suitable for the `set_schedule` method.
        """

        def run(T, steps):
            """Anneals a system at constant temperature and returns the state,
            energy, rate of acceptance, and rate of improvement."""
            E = self.energy()
            prevState = self.copy_state(self.state)
            prevEnergy = E
            accepts, improves = 0, 0
            for _ in range(steps):
                dE = self.move()
                if dE is None:
                    E = self.energy()
                    dE = E - prevEnergy
                else:
                    E = prevEnergy + dE
                if dE > 0.0 and math.exp(-dE / T) < random.random():
                    self.state = self.copy_state(prevState)
                    E = prevEnergy
                else:
                    accepts += 1
                    if dE < 0.0:
                        improves += 1
                    prevState = self.copy_state(self.state)
                    prevEnergy = E
            return E, float(accepts) / steps, float(improves) / steps

        step = 0
        self.start = time.time()

        # Attempting automatic simulated anneal...
        # Find an initial guess for temperature
        T = 0.0
        E = self.energy()
        self.update(step, T, E, None, None)
        while T == 0.0:
            step += 1
            dE = self.move()
            if dE is None:
                dE = self.energy() - E
            T = abs(dE)

        # Search for Tmax - a temperature that gives 98% acceptance
        E, acceptance, improvement = run(T, steps)

        step += steps
        while acceptance > 0.98:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        while acceptance < 0.98:
            T = round_figures(T * 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmax = T

        # Search for Tmin - a temperature that gives 0% improvement
        while improvement > 10.0:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmin = T

        # Calculate anneal duration
        elapsed = time.time() - self.start
        duration = round_figures(int(60.0 * minutes * step / elapsed), 2)

        # Don't perform anneal, just return params
        return {'tmax': Tmax, 'tmin': Tmin, 'steps': duration, 'updates': self.updates}


#================================================================
# Cohesiveness Wrapper
#================================================================
class MostCohesiveCirtuisGivenNCandidates(Annealer):
    def __init__(self, BiasDF, state, Graph, CandidateNodes, ExpDict, WeightDict, Weighted=True, Direction=False, minbias = -1, cohe_method='z'):
        self.BiasDF = BiasDF
        self.Graph = Graph
        self.CandidateNodes = CandidateNodes
        self.ExpDict = ExpDict
        self.WeightDict = WeightDict
        self.Weighted = Weighted
        self.Direction = Direction
        self.minbias = minbias
        self.cohe_method = cohe_method
        super(MostCohesiveCirtuisGivenNCandidates, self).__init__(state)
    def move(self):
        initial_energy = self.energy()
        idx_in = np.where(self.state==1)[0]
        idx_out = np.where(self.state==0)[0]
        idx_change_i = np.random.choice(idx_in, 1)
        idx_change_j = np.random.choice(idx_out, 1)
        self.state[idx_change_i] = 1 - self.state[idx_change_i]
        self.state[idx_change_j] = 1 - self.state[idx_change_j]
        strs = self.CandidateNodes[np.where(self.state==1)]
        if self.BiasDF.loc[strs, "EFFECT"].mean() < self.minbias:
            self.state[idx_change_i] = 1 - self.state[idx_change_i]
            self.state[idx_change_j] = 1 - self.state[idx_change_j]
        return (self.energy() - initial_energy)
    def energy(self):
        InCirtuitNodes = self.CandidateNodes[np.where(self.state==1)[0]]
        top_nodes = self.Graph.vs.select(label_in=InCirtuitNodes)
        g2 = self.Graph.copy()
        g2 = g2.subgraph(top_nodes)
        cohesives = []
        for v in g2.vs:
            coh, ITR = CohesivenessSingleNodeMaxInOut(self.Graph, g2, v["label"], self.WeightDict, weighted=self.Weighted)
            if self.cohe_method == 'vanilla':
                coh = coh
            # coh as zscore to exp
            elif self.cohe_method == 'z':
                exp_coh = self.ExpDict[v["label"]]
                diff = coh - np.mean(exp_coh)
                z = diff / np.std(exp_coh)
                coh = z
            elif self.cohe_method == 'ratio':
                exp_coh = self.ExpDict[v["label"]]
                ratio = coh / np.mean(exp_coh)
                coh = ratio
            cohesives.append(coh)
        cohesive = np.mean(cohesives)
        return 1 - cohesive


def CohesivenessSingleNodeMaxInOut(g, g_, STR, weightDict, weighted=True):
    Whole_EdgeList = []
    Cir_EdgeList = []
    Node = g.vs.find(label=STR)
    CircuitLables = set(g_.vs["label"])
    Total_Out,Circuit_Out = 0,0
    Total_In, Circuit_In = 0,0
    for _node in Node.successors():
        if weighted:
            weight = weightDict["{}-{}".format(STR, _node["label"])]
        else:
            weight = 1
        Total_Out += weight
        if _node["label"] in CircuitLables:
            Circuit_Out += weight
    for _node in Node.predecessors():
        if weighted:
            weight = weightDict["{}-{}".format(_node["label"], STR)]
        else:
            weight = 1
        Total_In += weight
        if _node["label"] in CircuitLables:
            Circuit_In += weight
    if (Total_In+Total_Out) == 0:
        return 0, None
    else:
        return ((Circuit_In+Circuit_Out)/(Total_In+Total_Out)), None
