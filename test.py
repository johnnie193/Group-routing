import torch
import utils

def reward(static, tour_indices):
    """
    Euclidean distance between all cities / nodes given by tour_indices
    """

    # print(tour_indices.size())

    # # Convert the indices back into a tour
    # idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    # tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # # Ensure we're always returning to the depot - note the extra concat
    # # won't add any extra loss, as the euclidean distance between consecutive
    # # points is 0
    # start = static.data[:, :, 0].unsqueeze(1)
    # y = torch.cat((start, tour, start), dim=1)

    # # Euclidean distance between each consecutive point
    # tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    # return tour_len.sum(1)
    # static 256 2 21 tour_indices 256 21
    # 先提取点
    # 扩展 indices(decoder_input:[256,21]) 以便在 dim=2 上进行 gather 操作
    device = static.device
    indices_expanded = tour_indices.unsqueeze(1).expand(-1, 2, -1)  # [256, 2, 21]

    # 使用 gather 在 dim=2 上进行操作，结果形状为 [256, 2, 21]
    ptr1_gathered = torch.gather(static, 2, indices_expanded)

    # 扩展 indices(torch.arrange:[256,21]) 以便在 dim=2 上进行 gather 操作
    default_indices = torch.arange(0, tour_indices.size(1)).unsqueeze(0).expand(tour_indices.size(0), -1).to(device)
    indices_expanded = default_indices.unsqueeze(1).expand(-1, 2, -1)  # [256, 2, 21]

    # 使用 gather 在 dim=2 上进行操作，结果形状为 [256, 2, 21]
    ptr2_gathered = torch.gather(static, 2, indices_expanded)
    print(ptr1_gathered)
    print(ptr2_gathered)
    # 计算两个张量之间的距离
    # point_distances = torch.sqrt(torch.sum((ptr1_gathered - ptr2_gathered) ** 2, dim=2))
    absolute_differences = torch.abs(ptr1_gathered - ptr2_gathered)
    # 对每个 dim0 的所有绝对差求和
    batch_distances_sum = torch.sum(absolute_differences, dim=(1, 2))
    print(batch_distances_sum)
    # 沿着点的维度求和
    # batch_distances_sum = torch.sum(point_distances, dim=1)
    # circles_sum = 2 * utils.find_num_circles(tour_indices)
    return batch_distances_sum

static = torch.Tensor([[[1,1,0],[2,3,3]],[[1,1,0],[2,3,3]]])
tour_indices = torch.Tensor([[1,2,0],[1,2,0]])
tour_indices = torch.tensor(tour_indices, dtype=torch.int64)
# 1 + 1 + 更好2
print(reward(static,tour_indices))