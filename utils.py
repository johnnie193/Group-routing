import torch
import datetime

def apply_batch_permutation_pytorch(permutations, sequences):
    start_time = datetime.datetime.now()
    # 创建一个拷贝，避免直接修改输入张量
    results = sequences

    # 获取批次大小
    batch_size = sequences.size(0)

    # 对每个样本应用其对应的置换索引
    for i in range(batch_size):
        # 获取当前样本的置换索引
        idx1 = (sequences[i] == permutations[i, 0]).nonzero(as_tuple=True)[0].item()
        idx2 = (sequences[i] == permutations[i, 1]).nonzero(as_tuple=True)[0].item()
        # 交换对应位置的元素
        results[i, idx1], results[i, idx2] = results[i, idx2], results[i, idx1].clone()

    end_time = datetime.datetime.now()
    delta_time = end_time - start_time
    delta_time = delta_time.seconds + delta_time.microseconds / 1000000.0
    # print("Time cost: {} ".format(delta_time))

    return results


from sympy.combinatorics import Permutation


def apply_batch_permutation_python(permutations, sequences):
    start_time = datetime.datetime.now()

    # 创建空列表，用于存储每个样本的置换结果
    results = torch.zeros(sequences.size())

    seq_len = sequences.size(1)
    # 对每个样本应用置换
    for i in range(sequences.size(0)):
        # 创建 SymPy 置换对象

        perm = Permutation(permutations[i][0],permutations[i][1])
        print(sequences[i])
        # 创建 SymPy 置换对象表示的序列
        seq_perm = Permutation(sequences[i].tolist())
        print(seq_len)
        print(seq_perm)
        print(perm)
        # 将结果转换为列表，并添加到结果列表中
        ans = torch.tensor([j^perm^seq_perm for j in range(seq_len)])
        results[i] = ans
        print(ans)

    end_time = datetime.datetime.now()
    delta_time = end_time - start_time
    delta_time = delta_time.seconds + delta_time.microseconds / 1000000.0
    print("Time cost: {} ".format(delta_time))

    return results


# permutations = torch.tensor([[2,4],[3,4]])
# sequences = torch.tensor([[0,1,2,3,4],[0,1,2,3,4]])
# print(sequences)
# print(apply_batch_permutation_python(permutations,sequences))
def find_num_circles(input):
    device = input.device
    num_circles = torch.zeros(input.size(0)).to(device)
    for i in range(input.size(0)):
        seq = input[i]
        num = 0
        mask = torch.zeros(input.size(1))
        for j in range(input.size(1)):
            if not mask[j]:
                cur = seq[j]
                mask[cur] = 1
                while cur != j:
                    mask[cur] = 1
                    cur = seq[cur]
                num = num + 1
        num_circles[i] = num
    return num_circles

i = torch.tensor([[1,0,2,3,4]])
print(find_num_circles(i))
