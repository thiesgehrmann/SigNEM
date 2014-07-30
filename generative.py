
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
  sigmoid(x)
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

def St_St1(Gamma, St, St_1, params):
  """
  St_St1(Gamma, St, St_1, params)
    Return the probability of transitioning from state St to state St-1

    Input:
      Gamma:  SigNEM Signals graph
      St:     A list of the states of each node in timepoint t
      St_1:   A list of the states of each node in timepoint t-1
      params: SigNEM parameters
    Output:
      P(St | St-1, \Theta)
  """

  probs = [];
  nsigs = len(Gamma);

  for j in xrange(nsigs):
    y   = St[j];
    paj = pa(Gamma, j);
    X   = [ St-1[p] for p in paj

    probs.append( activation(j, y, paj, X, params) );
  #efor

  P = np.prod(probs);

#edef

###############################################################################

###############################################################################

def activation(j, y, paj, X, params):
  """
  activation(j, y, paj, X, params)
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

  print g_self
  print g_paj
  print g_ppj

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
  emptyd(nsigs)
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
  pa(Gamma, j)
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

def psi_gen(observable, state):
  """

  """

  if x == 1:
    return np.random.beta(1, 10);
  else:
    return np.random.uniform(0,1);
  #fi
#edef


###############################################################################

A, B, C = [ 0, 1, 2 ];
Gamma, params = emptyg(3);

Gamma[A][B] = 1;
Gamma[A][C] = 1;

params[1][A][A] = 1;
params[1][B][B] = 1;
params[1][C][C] = 1;
params[1][A][B] = 1;
params[1][A][C] = 1;

params[2][B][A][B] = params[2][B][B][A] = 1;
params[2][C][A][C] = params[2][C][C][A] = 1;

