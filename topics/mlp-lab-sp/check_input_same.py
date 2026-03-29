# 检验同一个目录下input_dp1_pp1_tp1_mb4中0_step_input.pt
# input_dp4_pp1_tp1_mb4中0_step_input.pt 1_step_input.pt 2_step_input.pt 3_step_input.pt 拼起来是否相同
import os
import torch
# 加载input_dp1_pp1_tp1_mb4中的0_step_input.pt
all_same=1
for step in range(1000):
    input_file_1 = f"input_dp1_pp1_tp1_mb4/0_{step}_input.pt"
    data_1= torch.load(input_file_1, map_location='cpu')["inputs"]
    input_file_4_list = [f"input_dp4_pp1_tp1_mb4/{i}_{step}_input.pt" for i in range(4)]
    data_4_list = [torch.load(f, map_location='cpu')["inputs"] for f in input_file_4_list]
    data_4 = torch.cat(data_4_list, dim=0)
    if not torch.equal(data_1, data_4):
        all_same=0
        print(f"Step {step}: ❌ Inputs don't match")
        print(f"  input_dp1 shape: {data_1.shape}, input_dp4 shape: {data_4.shape}")
if all_same:
    print("✅ All inputs match")
all_same=1
for step in range(1000):
    input_file_1 = f"input_dp1_pp1_tp1_mb4/0_{step}_target.pt"
    data_1= torch.load(input_file_1, map_location='cpu')["target"]
    input_file_4_list = [f"input_dp4_pp1_tp1_mb4/{i}_{step}_target.pt" for i in range(4)]
    data_4_list = [torch.load(f, map_location='cpu')["target"] for f in input_file_4_list]
    data_4 = torch.cat(data_4_list, dim=0)
    if not torch.equal(data_1, data_4):
        all_same=0
        print(f"Step {step}: ❌ target don't match")
        print(f"  target_dp1 shape: {data_1.shape}, target_dp4 shape: {data_4.shape}")
if all_same:
    print("✅ All targets match")