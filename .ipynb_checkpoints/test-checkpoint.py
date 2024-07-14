from sympy.combinatorics import Permutation

# p = Permutation([0,1,2,3,4])
# q = Permutation(2,4)
# # [2, 0, 1]
# # Permutation(0, 2, 1)，注意0,2,1没有方括号
#
# print([i^p^q for i in range(5)])
# # [0, 2, 1]
import torch
input = torch.tensor([1,233,3232,3232])
input = input.repeat(5,1)
print(input)
input[2][:] = torch.tensor([1,3,4,5])
print(input)