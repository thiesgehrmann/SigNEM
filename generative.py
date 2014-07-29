
import numpy as np;

###############################################################################

def logistic(x):
  return 1.0 / (1.0 + np.exp(-x));
#edef

###############################################################################

def activation(j, y, paj, X, params):
  # j = node id j
  # y = node j value in time t
  # paj = list of parents of j
  # X = values of parents in time t-1
  # params = parameters
  #  params[0] = single factor parameters
  #  params[1] = edge factor parameters
  #  params[2] = pairwise factor parameters (??)

  # P(S_t^j = y | S_{t-1}^{pa(j)} = x, \theta_j) = 
  #  y * f(S_{t-1}^{pa(j)}) + (1-y)(1 - f(S_{t-1}^{pa(j)}))
  #
  # f(x) = 1 / ( 1 + exp(g(x)))
  # g(x) =  single factor + edge factors + pairwise factors

  npa = len(paj);

  g_self = params[0][j];
  g_paj  = sum([params[1][p][j] * px for (p, px) in zip(paj, X)]);
  g_ppj  = sum([ sum([params[2][j][paj[h]][paj[i]] * X[paj[h]] * X[paj[i]] for i in xrange(h+1, npa) ]) for h in xrange(0, npa-1) ]);

  print g_self
  print g_paj
  print g_ppj

  g = g_self + g_paj + g_ppj;

  logit_g = logistic(g);

  activ = y * logit_g + (1-y) * (1-logit_g);

  return activ;

#edef

###############################################################################

def emptyd(nsigs):

  Gamma = np.identity(nsigs, dtype=np.int8);

  params = [ [], [], [] ];
    # single node parameters
  params[0] = np.zeros(nsigs);
  params[1] = np.zeros((nsigs, nsigs));
  params[2] = np.zeros((nsigs, nsigs, nsigs));

  return Gamma, params;
#edef

###############################################################################

def pa(Gamma, j):

  parents = [ i for i in xrange(len(Gamma)) if Gamma[i][j] ];

  return parents;

#edef

###############################################################################

A, B, C = [ 0, 1, 2 ];
Gamma, params = emptyd(3);

Gamma[A][B] = 1;
Gamma[A][C] = 1;

params[1][A][A] = 1;
params[1][B][B] = 1;
params[1][C][C] = 1;
params[1][A][B] = 1;
params[1][A][C] = 1;

params[2][B][A][B] = params[2][B][B][A] = 1;
params[2][C][A][C] = params[2][C][C][A] = 1;

