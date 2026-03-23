import torch
import numpy as np


@torch.no_grad()
def analyze_attention(model, dataloader, device='cuda'):
    model.eval()
    attn_list = []

    for x, _ in dataloader:  # x: (B, 17, C) or (B, T, 17, C)
        if x.dim() == 4:  # (B,T,17,C)
            x = x[:, -1]  # 取最后一帧或你想要的帧
        x = x.to(device)

        # 钩子抓取注意力权重
        def hook(module, input, output):
            attn = module.attn_drop.softmax(-1)  # 去掉 dropout 的版本
            attn_list.append(attn.cpu())  # (B, H, 17, 23)

        handle = model.attn.register_forward_hook(hook)
        _ = model(x)  # 随便前向一次
        handle.remove()
        break  # 只看一个 batch 就够统计了（或去掉循环全部统计）

    attn = torch.cat(attn_list, dim=0)  # (Total_B, H, 17, 23)
    attn = attn.mean(1)  # 多头平均 → (B_total, 17, 23)
    attn17 = attn[:, :, :17]  # 只看 L1 部分，即关节→关节注意力

    # 关节名称（标准 Human3.6M / COCO 顺序）
    names = ["Hip", "RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot",
             "Spine", "Thorax", "Neck", "Head",
             "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist"]

    # 1. 部位分组
    part_dict = {
        "spine": [0, 7, 8, 9, 10],
        "left_arm": [11, 12, 13],
        "right_arm": [14, 15, 16],
        "left_leg": [4, 5, 6],
        "right_leg": [1, 2, 3]
    }

    # 这里可以直接复刻你贴的全部统计...
    # 例如：跨关节最强连接
    attn_flat = attn17.mean(0)  # (17,17)
    np_att = attn_flat.numpy()
    np_att[np.eye(17).astype(bool)] = 0  # 对角线置0
    idx = np.unravel_index(np.argsort(np_att.ravel())[-10:], (17, 17))
    print("Top-10 跨关节连接:")
    for i in range(10):
        q_id, k_id = idx[0][-i - 1], idx[1][-i - 1]
        print(f"{names[q_id]} ↔ {names[k_id]} : {np_att[q_id, k_id]:.4f}")

    # 其余部位内/跨部位、对角线协调、长短程比例... 都可以照着写 100 行以内搞定