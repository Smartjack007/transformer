import torch
import math


'''
Q = [
  [ 0.4532, -0.5358, -1.2009,  2.3768],   # q_a
  [ 3.1571,  1.9711, -1.3250,  1.6082],   # q_b
  [ 1.7140,  0.4084, -2.5555,  1.4152]    # q_c
]

K = [
  [-2.6898, -0.7520,  1.6816, -0.3854],   # k_a
  [-0.1974,  0.1310,  0.4958, -1.6271],   # k_b
  [-0.4883,  0.4634, -0.3311, -0.9610]    # k_c
]

scores[0][0] = q_a · k_a
= 0.4532*(-2.6898)
+ (-0.5358)*(-0.7520)
+ (-1.2009)*(1.6816)
+ 2.3768*(-0.3854)
= -1.2190 + 0.4029 - 2.0194 - 0.9160
≈ -3.7515

scores[0][1] = q_a · k_b
= 0.4532*(-0.1974)
+ (-0.5358)*(0.1310)
+ (-1.2009)*(0.4958)
+ 2.3768*(-1.6271)
= -0.0894 - 0.0702 - 0.5950 - 3.8663
≈ -4.6209

scores[0][2] = q_a · k_c
= 0.4532*(-0.4883)
+ (-0.5358)*(0.4634)
+ (-1.2009)*(-0.3311)
+ 2.3768*(-0.9610)
= -0.2213 - 0.2483 + 0.3976 - 2.2845
≈ -2.3565

scores[0] = [-3.7515, -4.6209, -2.3565]

Similarly:
scores[1] = [-12.8202, -3.6385, -1.7350]
scores[2] = [-9.7610, -3.8551, -1.1605]

scores = [
  [ -3.7515,  -4.6209, -2.3565],
  [-12.8202,  -3.6385, -1.7350],
  [ -9.7610,  -3.8551, -1.1605]
]

'''
def calculateScores(Q, K):
    """
    Q: (n, d)
    K: (n, d)

    return:
        scores: (n, n), where scores[i][j] = dot(Q[i], K[j])
    """
    n = Q.shape[0]  # number of tokens
    scores = torch.zeros(n, n)

    for i in range(n):
        for j in range(n):
            scores[i][j] = torch.dot(Q[i], K[j])

    return scores

'''
there are three tokens , each row represents a token 

[1.0, 0.0, 1.0, 0.0] represents token a 
[0.0, 1.0, 0.0, 1.0] represents token b
[1.0, 1.0, 0.0, 0.0] represents token c

'''
X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0]
])

print("X =", X)
'''
can be genrated automatically
W_q = torch.randn(4, 4)
W_k = torch.randn(4, 4)
W_v = torch.randn(4, 4)
'''

W_q = torch.tensor([
      [-0.3168, -0.9835, -1.1622,  0.8748],
      [ 2.0308,  1.3919, -1.3933,  0.5404],
      [ 0.7700,  0.4477, -0.0387,  1.5020],
      [ 1.1263,  0.5792,  0.0683,  1.0678]
])

W_k = torch.tensor([
      [-1.1980, -0.7275,  0.5288,  0.2358],
      [ 0.7097,  1.1909, -0.8599, -1.1968],
      [-1.4918, -0.0245,  1.1528, -0.6212],
        [-0.9071, -1.0599,  1.3557, -0.4303]
])

W_v = torch.tensor([
      [-1.0767, -1.6545,  0.1472,  0.5024],
      [ 0.5783, -1.2717, -0.3273,  1.1946],
      [ 0.7919, -0.0733,  0.3094, -0.1051],
      [-0.9542, -0.1368, -2.2398, -0.0722]
])


# Q = X @ W_q means:
# each row in X is projected into a query vector.
#
# For example:
# q_a = x_a @ W_q
#
# q_a[0] is computed by:
# x_a dot the 1st column of W_q
# x_a = [1.0, 0.0, 1.0, 0.0]
# 1st column of W_q = [-0.3168, 2.0308, 0.7700,1.1263]
# = 1*(-0.3168) + 0*(2.0308) + 1*(0.7700) + 0*(1.1263)
# = 0.4532
#
# Similarly:
# q_a[1] = x_a dot the 2nd column of W_q = -0.5358
# q_a[2] = x_a dot the 3rd column of W_q = -1.2009
# q_a[3] = x_a dot the 4th column of W_q = 2.3768
#
# So:
# q_a = [0.4532, -0.5358, -1.2009, 2.3768]
#
# In the same way:
# q_b = [3.1571, 1.9711, -1.3250, 1.6082]
# q_c = [1.7140, 0.4084, -2.5555, 1.4152]
#
# Q = tensor([[ 0.4532, -0.5358, -1.2009,  2.3768],
#        [ 3.1571,  1.9711, -1.3250,  1.6082],
#        [ 1.7140,  0.4084, -2.5555,  1.4152]])

Q = X @ W_q
K = X @ W_k
V = X @ W_v

'''
Q = tensor([[ 0.4532, -0.5358, -1.2009,  2.3768],
        [ 3.1571,  1.9711, -1.3250,  1.6082],
        [ 1.7140,  0.4084, -2.5555,  1.4152]])
        
K = tensor([[-2.6898, -0.7520,  1.6816, -0.3854],
        [-0.1974,  0.1310,  0.4958, -1.6271],
        [-0.4883,  0.4634, -0.3311, -0.9610]])
        
V = tensor([[-0.2848, -1.7278,  0.4566,  0.3973],
        [-0.3759, -1.4085, -2.5671,  1.1224],
        [-0.4984, -2.9262, -0.1801,  1.6970]])        
'''
print("Q =", Q)
print("K =", K)
print("V =", V)

# we can use tradicitonal approach to calculate scores
# pelase see calculateScores
# the result is as the same as  scores = Q @ K.T
scores = Q @ K.T
print("scores=", V)


weights = torch.softmax(scores / math.sqrt(4), dim=-1)
print("weights =", weights)


#weights = tensor([
#    [0.2735, 0.1770, 0.5495], 
#    [0.0028, 0.2777, 0.7195],
#    [0.0107, 0.2042, 0.7851]
#])

# Each row represents a query token.
# Each value in the row indicates how much attention
# the query token pays to each key token (including itself).
# The values in each row sum to 1 (softmax).

# For example:
# [0.2735, 0.1770, 0.5495]

# token A attends to:
# - token A: 27%
# - token B: 17%
# - token C: 55%

# This means token A focuses mostly on token C.

output = weights @ V
print("output =", output)