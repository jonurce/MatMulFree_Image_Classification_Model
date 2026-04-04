
import torch


if __name__ == "__main__":
    tensor_f32 = torch.randn(4, 4, device="cuda", dtype=torch.float32)
    tensor_f8 = tensor_f32.to(torch.float8_e4m3fn) 
    tensor_f8_in_f32 = tensor_f8.to(torch.float32) 
    print("f32 tensor:", tensor_f32)
    print("f8 tensor:", tensor_f8)
    print("f8 tensor in f32:", tensor_f8_in_f32)

    print("f32 first value with all decimals:", tensor_f32[0, 0].item())
    print("f8 first value with all decimals:", tensor_f8[0, 0].item())
    print("f8 first value in f32 with all decimals:", tensor_f8_in_f32[0, 0].item())
    