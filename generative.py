
import numpy as np;

###############################################################################
#  Common variables:                                                          # 
#                                                                             #
#   nsigs: The number of signal nodes in the NEM                              #
#     type: int                                                               #
#                                                                             #
#   Gamma: The Signals graph of the NEM                                       #
#     type: np.matrix((nsigs, nsigs), dtype=np.int8)                          #
#                                                                             #
#   params: Parameters for the signem                                         #
#     params[0] = single factor parameters                                    #
#       type: np.zeros(nsigs, dtype=np.float)                                 #
#     params[1] = edge factor parameters                                      #
#       type: np.zeros((nsigs, nsigs), dtype=np.float);                       #
#     params[2] = pairwise factor parameters (??)                             #
#       type: np.zeros((nsigs, nsigs, nsigs), dtype=np.float);                #
#                                                                             #
###############################################################################

###############################################################################

def sigmoid(x):
  """
  y = sigmoid(x)
    Return the value of the sigmoid function at location x

    Input:
      x: location on the x-axis

    Output:
      y: Value of sigmoid at location x
  """
  y = 1.0 / (1.0 + np.exp(-float(x)));
  return y;
#edef

###############################################################################

#def POSH(Gamma, params, S, O):
#
#
#
##edef

###############################################################################

def St_St1(Gamma, params, St, St_1):
  """
  P = St_St1(Gamma, St, St_1, params)
    Return the probability of transitioning from state St to state St-1

    Input:
      Gamma:  SigNEM Signals graph
      params: SigNEM parameters
      St:     A list of the states of each node in timepoint t
      St_1:   A list of the states of each node in timepoint t-1
    Output:
      P: Probability of state sequence between St and St-1
        P(S_{t} | S_{t-1}, \Theta)
  """

  probs = [];
  nsigs = len(Gamma);

  for j in xrange(nsigs):
    y   = St[j];
    paj = pa(Gamma, j);
    X   = [ St_1[p] for p in paj ];

    probs.append( activation(j, y, paj, X, params) );
  #efor

  P = np.prod(probs);

  return P;

#edef

###############################################################################

def P_S(Gamma, params, S, prior_S1):
  """
  P = P_S(Gamma, params, S, prior_S1)
    Probability of state sequence

    Input:
      Gamma:    SigNEM signals graph
      params:   SigNEM parameters
      St:       sequence of states
      prior_S1: The prior for the initial state

    Output:
      P: Probability of state sequence
       prior_S1 * \prod_{t=2}^{T} P(S_t | S_{t-1}, \Theta)

  """

  nsigs = len(Gamma);
  ntime = len(S);

  probs = [ prior_S1 ];

  for t in xrange(1, ntime):
    probs.append( St_St1(Gamma, params, S[t], S[t-1]) );
  #efor

  P = np.prod(probs);

  return P;

#edef

###############################################################################

def activation(j, y, paj, X, params):
  """
  p_y_given_x = activation(j, y, paj, X, params)
    Return the activation of node j at timepoint t given the state of it's parents in timepoint t-1.

    Input:
      j:      Node id of a node in Signals Graph
      y:      The state of node j
      paj:    The parents of j
      X:      The states of the parents of j
      params: The parameters for the SigNEM

    Output:
      p_y_given_x: The probability of node j having state y, given the parent states in previous timepoint

        P(S_t^j = y | S_{t-1}^{pa(j)} = x, \theta_j) = 
           y * f(S_{t-1}^{pa(j)}) + (1-y)(1 - f(S_{t-1}^{pa(j)}))
       where,
        f(x) = 1 / ( 1 + exp(g(x)))
        g(x) =  single factor + edge factors + pairwise factors (factorization of graph)
  """
  npa = len(paj);

  g_self = params[0][j];
  g_paj  = sum([params[1][p][j] * px for (p, px) in zip(paj, X)]);
  g_ppj  = sum([ sum([params[2][j][paj[h]][paj[i]] * X[paj[h]] * X[paj[i]] for i in xrange(h+1, npa) ]) for h in xrange(0, npa-1) ]);

  g = g_self + g_paj + g_ppj;

  sigmoid_g = sigmoid(g);

  if y == 1:
    p_y_given_x = sigmoid_g;
  else:
    p_y_given_x = 1 - sigmoid_g;
  #fi

  return p_y_given_x;

#edef

###############################################################################

def emptyg(nsigs):
  """
  Gamma, params = emptyg(nsigs)
    Return an empty data structure for the inputs to the generative model

    Input:
      nsigs: The number of signal nodes in the signals graph

    Output:
      Gamma:  An empty identity matrix for the signals graph
      params: An empty structure of parameters for the SigNEM
  """

  Gamma = np.identity(nsigs, dtype=np.int8);

  params = [ [], [], [] ];
    # single factor parameters
  params[0] = np.zeros(nsigs);
    # Edge factor parameters
  params[1] = np.zeros((nsigs, nsigs));
    # Pairwise factor parameters
  params[2] = np.zeros((nsigs, nsigs, nsigs));

  return Gamma, params;
#edef

###############################################################################

def pa(Gamma, j):
  """
  parents = pa(Gamma, j)
    Retrieve a list of the parents of node j in graph Gamma

    Input:
      Gamma: SigNEM Signals Graph
      j:     Node id

    Output:
      parents: A list of the parents of node j

  """

  parents = [ i for i in xrange(len(Gamma)) if Gamma[i][j] ];
  return parents;
#edef

###############################################################################

def psi_gen(state):
  """
  psi = psi_gen(state)
    Generate an observable point from a distribution based on the state of parent node

    Input:
      state: The state of a given node

    Output:
      psi: A sampled probability from a beta(1,10) distribution if state==1, and uniform otherwise
  """


  if state == 1:
    psi = np.random.beta(1, 10);
  else:
    psi = np.random.uniform(0,1);
  #fi

  return psi;

#edef

###############################################################################

def o_gen(state, mu0=0, stdev0=1, mu1=3, stdev1=1):

  if state == 0:
    O = np.random.normal(mu0, stdev0);
  else:
    O = np.random.normal(mu1, stdev1);
  #fi

  return O;

#edef

###############################################################################

def gen_observables(K, S):
  """
  O = gen_observables(K, S)
    Generate log-odds data for the SigNEM model

    Inputs:
      K:   The number of observables to generate data for.
      S:   The state sequences of each signal in the SigNEM graph.
        type: np.matrix((T, nsigs))

    Output:
      O: The log-odds of attachment for each observable in each time point.
        type: np,matrix((K, T));
  """

  T, nsigs = S.shape;

  O = np.zeros((K, T));

  if K < nsigs:
    print "Error, K must be >= nsigs"
    return None;
  #fi

    # We first assign at least one observable to each signal node
  assignments = [ j for j in xrange(nsigs) ];
    # And then the rest randomly
  assignments.extend( np.random.randint(nsigs, size=(K - nsigs)) );

  St = S.transpose();

  for (k,j) in enumerate(assignments):
    
    for t in xrange(T):
      O[k,t] = o_gen(S[t,j]) #????
  #efor
    
  return O;

#edef

###############################################################################

def gen_state(Gamma, params, St, return_probs=False):
  """
  St1, [P] = gen_state(Gamma, params, St)
    Generate the state of the next time point based on the current one.

    Input:
      Gamma:  SigNEM Signals graph
      params: SigNEM parameters
      St:     A list of the states of each node in timepoint t
      rprobs: Return the probabilities of activation at each timepoint?
              If yes, then P is also returned.

    Output:
      St1: The state in timepoint t+1
      P:   The probability of this activation of this node
  """

  nsigs = len(Gamma);
  St1   = np.zeros(nsigs);
  P     = np.zeros(nsigs);

  for j in xrange(nsigs):
    y   = 1;
    paj = pa(Gamma, j);
    X   = St;
    
    p_1 = activation(j, y, paj, X, params);

      # Or, alternatively
    if p_1 > np.random.uniform(0,1):
    #if p_1 <= 0.5:
      y = 0;
    #fi
    St1[j] = y;
    P[j] = p_1;
  #efor

  return St1, P;
#edef

###############################################################################

def gen_state_sequence(Gamma, params, S1, T, rprobs=False):
  """
  S = gen_state_sequence(Gamma, params, S1, T):
    Generate, from an initial state S1, a state sequence with T time points

    Input:
      Gamma:  SigNEM Signals graph
      params: SigNEM parameters
      S1:     The initial state configuration
      T:      The number of timepoints to generate
      rprobs: Return the activation probabilities at each tiempoint?
              If yes, then P is also returned.

    Output:
      S:   The state sequence
      [P]: The activation probabilities of each node at each timepoint
  """

  S = [ np.array(S1, dtype=np.float) ];
  P = [];

  while len(S) < T:
    Si, Pi = gen_state(Gamma, params, S[-1]);
    S.append(Si);
    P.append(Pi);
  #ewhile

  if rprobs:
    return np.matrix(S), np.matrix(P);
  else:
    return np.matrix(S);
  #fi

#edef

###############################################################################

def gen_data(Gamma, params, S1, T, K):
  """
  S, O = gen_data(Gamma, params, S1, T, K)
    Generate simulation data, based on:

    Inputs:
      Gamma:  SigNEM Signals graph
      params: SigNEM parameters
      S1:     The initial state configuration
      T:      The number of timepoints to generate
      K:      The number of observables to generate.

    Outputs: 
      S: State sequences for each node over time
        type: np.matrix((T, nsigs))
      O: Observations for each observable
        type: np.matrix((T, K));
  """

  S = gen_state_sequence(Gamma, params, S1, T);
  O = gen_observables(K, S);

  return S, O;

#edef


###############################################################################

###############################################################################
# Example
#
#  A --> B --| C
#  |           ^
#  _           |
#  D-----------+
#
#
#  Inhibit:  X --| Y
#  Activate: X --> Z
#
###############################################################################


A, B, C, D = [ 0, 1, 2, 3 ];
Gamma, params = emptyg(4);

Gamma[A, B] = 1;
Gamma[A, C] = 1;
Gamma[C, D] = 1;
Gamma[B, D] = 1;

params[1][A, A] = 0.1;
params[1][B, B] = 0.1;
params[1][C, C] = 0.1;
params[1][D, D] = 0.1;
params[1][A, B] = -1;
params[1][A, C] = 0.5;
params[1][C, D] = -0.8;
params[1][B, D] = 0.9;

params[2][B, A, B] = params[2][B, B, A] = 0.5;
params[2][C, A, C] = params[2][C, C, A] = 0.5;
params[2][D, D, C] = params[2][D, C, D] = 0.5;
params[2][D, B, C] = params[2][D, C, B] = 0.5;
params[2][D, B, D] = params[2][D, D, B] = 0.5;

S1 = [ 1, 0, 0, 0 ];

gen_data(Gamma, params, S1, 10, 30)
