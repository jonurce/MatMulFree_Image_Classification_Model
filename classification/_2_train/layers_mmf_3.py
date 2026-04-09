import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load


################################ MMF & MMFv2 ################################

################ MMF Linear Function ################
class MMFLinearFunction(torch.autograd.Function):
    # Theory: X ∈ R^{M×N}, W ∈ R^{N×K}, b ∈ R^K
    # Code: X [M,N], W [K, N], b [K] 
    # Step 1. Load from HBM: X, W, b (HBM slow)
    @staticmethod
    def forward(ctx, X, W, b):
        # Step 2. On chip: Y_tilde, mu, var, r <- rms_norm_fwd(X)
        mu   = X.mean(dim=-1, keepdim=True)             # [M, 1]
        var  = X.var(dim=-1, keepdim=True, unbiased=False) # [M, 1]
        r    = 1.0 / (var + 1e-8).sqrt()                 # [M, 1]

        y_norm = r * (X - mu)                           # [M, N]

        # Inside Step 2: activation_quant
        s_act  = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # [M, 1]
        y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act     # [M, N]

        # Step 3: On chip: W_tilde <- weight_quant(W)
        s_w     = 1.0 / W.abs().mean().clamp(min=1e-8) # scalar
        
        # w_tilde [K, N] -> ternary: {-mean(|W|), 0, +mean(|W|)} = {-1/s_w, 0, +1/s_w}
        # w_tilde = (s_w * W).round().clamp(-1, 1) / s_w # no gradient flows through round/clamp
        w_tilde = W - (W - (s_w * W).round().clamp(-1, 1) / s_w).detach() # STE allows gradient to flow through to W

        # Step 4: On chip: O <- Y_tilde ⊛ W_tilde + b
        O = y_tilde @ w_tilde.t() + b    # [M, K] <- [M, N] @ [N, K] + [K]

        # Step 5: Store to HBM: O (return) + mu, var, r + X, W, b
        ctx.save_for_backward(X, W, b, mu, var, r)

        return O # [M, K]

    # Step 1. Load from HBM: X, W, b, O (not used), mu, var, r, dO (HBM slow)
    @staticmethod
    def backward(ctx, dO):
        X, W, b, mu, var, r = ctx.saved_tensors

        # Step 2. On chip: dY <- dO × W^T
        dY = dO @ W # [M, N] <- [M, K] @ [K, N] 

        # Step 3. On chip: dX, Y_tilde <- rms_norm_bwd(dY, X, mu, var, r)
        y_norm  = r * (X - mu)
        s_act   = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        Y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act

        dsigma2 = (dY * (X - mu) * -0.5 * r**3).sum(dim=-1, keepdim=True)  # [M, 1]

        dmu = (-r * dY).sum(dim=-1, keepdim=True) + dsigma2 * (X - mu).mean(dim=-1, keepdim=True)  # [M, 1]
        
        N  = X.shape[-1]
        dX = r * dY + 2 * dsigma2 * (X - mu) / N + dmu / N

        # Step 4: On chip: dW <- dO^T × Y_tilde
        dW = dO.t() @ Y_tilde   # [K, N] = [K, M] @ [M, N] 

        # Step 5: On chip: db <- sum(dO)
        db = dO.sum(dim=0)    # [K]

        # Step 6: Store dX, dW, db to HBM
        # Same args as forward(ctx, X, W, b)
        return dX, dW, db

################ MMF Linear Layer ################
class MMFLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        in_features = int(in_features)
        out_features = int(out_features)

        # X [M,N] * W.t() [N,K] = O [M,K]
        self.weight = nn.Parameter(torch.randn(out_features, in_features)) # [K, N]
        self.bias = nn.Parameter(torch.zeros(out_features)) # [K]

    def forward(self, x):
        return MMFLinearFunction.apply(x, self.weight, self.bias)

################ MMF Conv2d Function ################
class MMFConv2dFunction(torch.autograd.Function):
    # Theory: X ∈ R^{B×C_in×H_in×W_in}, W ∈ R^{C_out×C_in×kH×kW}, b ∈ R^{C_out}
    # Code: X [B, C_in, H_in, W_in], W [C_out, C_in, kH, kW], b [C_out]
    ### Step 1. Load from HBM: X, W, b (HBM slow)
    @staticmethod
    def forward(ctx, X, W, b, stride, padding):

        # Before nothing, get shapes
        B, C_in, H_in, W_in = X.shape

        # Normalize over C_in per spatial position — reshape to [B*H_in*W_in, C_in]
        # so that dim=-1 normalization operates over C_in exactly as in Linear
        X_flat = X.permute(0, 2, 3, 1).reshape(-1, C_in)  # [M, C_in] where M=B*H_in*W_in

        ### Step 2. On chip: Y_tilde, mu, var, r <- rms_norm_fwd(X)
        mu   = X_flat.mean(dim=-1, keepdim=True)             # [M, 1]
        var  = X_flat.var(dim=-1, keepdim=True, unbiased=False) # [M, 1]
        r    = 1.0 / (var + 1e-8).sqrt()                 # [M, 1]

        y_norm = r * (X_flat - mu)                           # [M, C_in]

        # Inside Step 2: activation_quant
        s_act  = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # [M, 1]
        y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act     # [M, C_in]

        # Reshape y_tilde back to [B, C_in, H_in, W_in] for conv2d
        y_tilde_4d = y_tilde.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]


        ### Step 3: On chip: W_tilde <- weight_quant(W)
        s_w     = 1.0 / W.abs().mean().clamp(min=1e-8) # scalar
        
        # w_tilde [C_out, C_in, kH, kW] -> ternary: {-mean(|W|), 0, +mean(|W|)} = {-1/s_w, 0, +1/s_w}
        # w_tilde = (s_w * W).round().clamp(-1, 1) / s_w # no gradient flows through round/clamp
        w_tilde = W - (W - (s_w * W).round().clamp(-1, 1) / s_w).detach() # STE allows gradient to flow through to W

        ### Step 4: On chip: O <- Y_tilde ⊛ W_tilde + b
        O = F.conv2d(y_tilde_4d, w_tilde, b, stride, padding)  # [B, C_out, H_out, W_out]


        ### Step 5: Store to HBM: O (return) + mu, var, r + X, W, b
        ctx.save_for_backward(X, W, b, mu, var, r)
        ctx.stride = stride
        ctx.padding = padding

        return O # [B, C_out, H_out, W_out]

    ### Step 1. Load from HBM: X, W, b, O (not used), mu, var, r, dO (HBM slow)
    @staticmethod
    def backward(ctx, dO):
        X, W, b, mu, var, r = ctx.saved_tensors
        stride, padding = ctx.stride, ctx.padding

        B, C_in, H_in, W_in = X.shape

        ### Step 2. On chip: dY <- dO × W^T
        # In conv2d, W^T means transposed convolution
        dY_4d = torch.nn.grad.conv2d_input(X.shape, W, dO.float(), stride=stride, padding=padding)  # [B, C_in, H_in, W_in]

        # Reshape dY to [M, C_in] to match RMSNorm dimension: M = B * H_in * W_in
        dY = dY_4d.permute(0, 2, 3, 1).reshape(-1, C_in)  # [M, C_in]

        ### Step 3. On chip: dX, Y_tilde <- rms_norm_bwd(dY, X, mu, var, r)
        X_flat = X.permute(0, 2, 3, 1).reshape(-1, C_in)              # [M, C_in]

        y_norm  = r * (X_flat - mu)
        s_act   = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        Y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act # [M, C_in]

        dsigma2 = (dY * (X_flat - mu) * -0.5 * r**3).sum(dim=-1, keepdim=True)  # [M, 1]

        dmu = (-r * dY).sum(dim=-1, keepdim=True) + dsigma2 * (X_flat - mu).mean(dim=-1, keepdim=True)  # [M, 1]
        
        # normalization dimension is C_in
        dX_flat = r * dY + 2 * dsigma2 * (X_flat - mu) / C_in + dmu / C_in # [M, C_in]

        # Reshape dX back to [B, C_in, H_in, W_in]
        dX = dX_flat.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]


        ### Step 4: On chip: dW <- dO^T × Y_tilde
        # Reshape Y_tilde back to 4d for conv2d_weight
        Y_tilde_4d = Y_tilde.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]
        dW = torch.nn.grad.conv2d_weight(Y_tilde_4d, W.shape, dO.float(), stride=stride, padding=padding)  # [C_out, C_in, kH, kW]

        ### Step 5: On chip: db <- sum(dO)
        db = dO.sum(dim=(0, 2, 3))    # [C_out]

        ### Step 6: Store dX, dW, db to HBM
        # Same args as forward(ctx, X, W, b, stride, padding)
        return dX, dW, db, None, None   # None for stride, padding

################ MMF Conv2d Layer ################
class MMFConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        in_channels = int(in_channels)
        out_channels = int(out_channels)
        
        # X [B, C_in, H_in, W_in] conv W [C_out, C_in, kH, kW] = O [B, C_out, H_out, W_out]

        # Weights: [C_out, C_in, kH, kW]
        self.weight  = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size)) 

        # Bias [C_out]
        self.bias    = nn.Parameter(torch.zeros(out_channels))

        # Stride and padding (scalars)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return MMFConv2dFunction.apply(x, self.weight, self.bias, self.stride, self.padding)





################ MMF Conv2d Residual Layer ################
class MMFConv2dRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # X [B, C_in, H_in, W_in] conv W [C_out, C_in, kH, kW] = O [B, C_out, H_out, W_out]

        # Weights: [C_out, C_in, kH, kW]
        self.weight  = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size)) 

        # Bias [C_out]
        self.bias    = nn.Parameter(torch.zeros(out_channels))

        # Stride and padding (scalars)
        self.stride = stride
        self.padding = padding

        # Residual projection: needed when input and output shapes differ
        # If in_channels == out_channels and stride == 1: identity (no projection needed)
        # If shapes differ: 1x1 conv to match channels and spatial size
        if in_channels != out_channels or stride != 1:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        else:
            self.projection = None  # identity skip — x passes through unchanged

    def forward(self, x):
    
        # MMF path: full ternary conv
        mmf_out = MMFConv2dFunction.apply(x, self.weight, self.bias, self.stride, self.padding)

        # Skip path: identity or projection to match shape
        if self.projection is not None:
            skip = self.projection(x)   # [B, C_out, H_out, W_out]
        else:
            skip = x  # [B, C_in, H_in, W_in] = [B, C_out, H_out, W_out]

        # Residual connection: F(x) + x
        return mmf_out + skip




################################ MMFv1 ################################
################ MMF Linear Layer Without Backward Quantization ################
class MMFLinearv1(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # X [M,N] * W.t() [N,K] = O [M,K]
        self.weight = nn.Parameter(torch.randn(out_features, in_features)) # [K, N]
        self.bias = nn.Parameter(torch.zeros(out_features)) # [K]

    def forward(self, x):
        # Step 2. On chip: Y_tilde, mu, var, r <- rms_norm_fwd(X)
        mu   = x.mean(dim=-1, keepdim=True)             # [M, 1]
        var  = x.var(dim=-1, keepdim=True, unbiased=False) # [M, 1]
        r    = 1.0 / (var + 1e-8).sqrt()                 # [M, 1]

        y_norm = r * (x - mu)                           # [M, N]

        # Inside Step 2: activation_quant
        s_act  = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # [M, 1]
        y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act     # [M, N]

        # Step 3: On chip: W_tilde <- weight_quant(W)
        W = self.weight # [K, N]
        s_w     = 1.0 / W.abs().mean().clamp(min=1e-8) # scalar
        
        # w_tilde [K, N] -> ternary: {-mean(|W|), 0, +mean(|W|)} = {-1/s_w, 0, +1/s_w}
        # w_tilde = (s_w * W).round().clamp(-1, 1) / s_w # no gradient flows through round/clamp
        # w_tilde = W - (W - (s_w * W).round().clamp(-1, 1) / s_w).detach() # STE allows gradient to flow through to W
        w_tilde = W + ((s_w * W).round().clamp(-1, 1) / s_w - W).detach()

        # Step 4: On chip: O <- Y_tilde ⊛ W_tilde + b
        return y_tilde @ w_tilde.t() + self.bias    # [M, K] <- [M, N] @ [N, K] + [K]

################ MMF Conv2d Layer Without Backward Quantization ################
class MMFConv2dv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # X [B, C_in, H_in, W_in] conv W [C_out, C_in, kH, kW] = O [B, C_out, H_out, W_out]

        # Weights: [C_out, C_in, kH, kW]
        self.weight  = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size)) 

        # Bias [C_out]
        self.bias    = nn.Parameter(torch.zeros(out_channels))

        # Stride and padding (scalars)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Before nothing, get shapes
        B, C_in, H_in, W_in = x.shape

        # Normalize over C_in per spatial position — reshape to [B*H_in*W_in, C_in]
        # so that dim=-1 normalization operates over C_in exactly as in Linear
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C_in)  # [M, C_in] where M=B*H_in*W_in

        ### Step 2. On chip: Y_tilde, mu, var, r <- rms_norm_fwd(x)
        mu   = x_flat.mean(dim=-1, keepdim=True)             # [M, 1]
        var  = x_flat.var(dim=-1, keepdim=True, unbiased=False) # [M, 1]
        r    = 1.0 / (var + 1e-8).sqrt()                 # [M, 1]

        y_norm = r * (x_flat - mu)                           # [M, C_in]

        # Inside Step 2: activation_quant
        s_act  = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # [M, 1]
        y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act     # [M, C_in]

        # Reshape y_tilde back to [B, C_in, H_in, W_in] for conv2d
        y_tilde_4d = y_tilde.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]
        # y_norm_4d = y_norm.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]


        ### Step 3: On chip: W_tilde <- weight_quant(W)
        W = self.weight
        s_w     = 1.0 / W.abs().mean().clamp(min=1e-8) # scalar
        
        # w_tilde [C_out, C_in, kH, kW] -> ternary: {-mean(|W|), 0, +mean(|W|)} = {-1/s_w, 0, +1/s_w}
        # w_tilde = (s_w * W).round().clamp(-1, 1) / s_w # no gradient flows through round/clamp
        # w_tilde = W - (W - (s_w * W).round().clamp(-1, 1) / s_w).detach() # STE allows gradient to flow through to W
        w_tilde = W + ((s_w * W).round().clamp(-1, 1) / s_w - W).detach()

        ### Step 4: On chip: O <- Y_tilde ⊛ W_tilde + b
        return F.conv2d(y_tilde_4d, w_tilde, self.bias, self.stride, self.padding)  # [B, C_out, H_out, W_out]






################################ MMFv3 ################################

################ MMF Linear Function with Activation Quantization to F8 ################
class MMFLinearFunctionv3(torch.autograd.Function):
    # Theory: X ∈ R^{M×N}, W ∈ R^{N×K}, b ∈ R^K
    # Code: X [M,N], W [K, N], b [K] 
    # Step 1. Load from HBM: X, W, b (HBM slow)
    @staticmethod
    def forward(ctx, X, W, b):
        # Step 2. On chip: Y_tilde, mu, var, r <- rms_norm_fwd(X)
        mu   = X.mean(dim=-1, keepdim=True)             # [M, 1]
        var  = X.var(dim=-1, keepdim=True, unbiased=False) # [M, 1]
        r    = 1.0 / (var + 1e-8).sqrt()                 # [M, 1]

        y_norm = r * (X - mu)                           # [M, N]

        # Inside Step 2: activation_quant to F8
        # E4M3 (High Precision): max value: 448 /vs/ E5M2 (High Range): max value: 57,344
        s_act  = 448.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # [M, 1]
        y_tilde = (s_act * y_norm).to(torch.float8_e4m3fn).to(torch.float32) / s_act     # [M, N]


        # Step 3: On chip: W_tilde <- weight_quant(W)
        s_w     = 1.0 / W.abs().mean().clamp(min=1e-8) # scalar
        
        # w_tilde [K, N] -> ternary: {-mean(|W|), 0, +mean(|W|)} = {-1/s_w, 0, +1/s_w}
        # w_tilde = (s_w * W).round().clamp(-1, 1) / s_w # no gradient flows through round/clamp
        w_tilde = W - (W - (s_w * W).round().clamp(-1, 1) / s_w).detach() # STE allows gradient to flow through to W

        # Step 4: On chip: O <- Y_tilde ⊛ W_tilde + b
        O = y_tilde @ w_tilde.t() + b    # [M, K] <- [M, N] @ [N, K] + [K]

        # Step 5: Store to HBM: O (return) + mu, var, r + X, W, b
        ctx.save_for_backward(X, W, b, mu, var, r)

        return O # [M, K]

    # Step 1. Load from HBM: X, W, b, O (not used), mu, var, r, dO (HBM slow)
    @staticmethod
    def backward(ctx, dO):
        X, W, b, mu, var, r = ctx.saved_tensors

        # Step 2. On chip: dY <- dO × W^T
        dY = dO @ W # [M, N] <- [M, K] @ [K, N] 

        # Step 3. On chip: dX, Y_tilde <- rms_norm_bwd(dY, X, mu, var, r) + activation_quant to F8
        y_norm  = r * (X - mu)

        # Inside Step 3: activation_quant to F8
        # E4M3 (High Precision): max value: 448 /vs/ E5M2 (High Range): max value: 57,344
        s_act  = 448.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        Y_tilde = (s_act * y_norm).to(torch.float8_e4m3fn).to(torch.float32) / s_act # [M, N]

        dsigma2 = (dY * (X - mu) * -0.5 * r**3).sum(dim=-1, keepdim=True)  # [M, 1]

        dmu = (-r * dY).sum(dim=-1, keepdim=True) + dsigma2 * (X - mu).mean(dim=-1, keepdim=True)  # [M, 1]
        
        N  = X.shape[-1]
        dX = r * dY + 2 * dsigma2 * (X - mu) / N + dmu / N

        # Step 4: On chip: dW <- dO^T × Y_tilde
        dW = dO.t() @ Y_tilde   # [K, N] = [K, M] @ [M, N] 

        # Step 5: On chip: db <- sum(dO)
        db = dO.sum(dim=0)    # [K]

        # Step 6: Store dX, dW, db to HBM
        # Same args as forward(ctx, X, W, b)
        return dX, dW, db

################ MMF Linear Layer ################
class MMFLinearv3(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        in_features = int(in_features)
        out_features = int(out_features)

        # X [M,N] * W.t() [N,K] = O [M,K]
        self.weight = nn.Parameter(torch.randn(out_features, in_features)) # [K, N]
        self.bias = nn.Parameter(torch.zeros(out_features)) # [K]

    def forward(self, x):
        return MMFLinearFunctionv3.apply(x, self.weight, self.bias)

################ MMF Conv2d Function with Activation Quantization to F8 ################
class MMFConv2dFunctionv3(torch.autograd.Function):
    # Theory: X ∈ R^{B×C_in×H_in×W_in}, W ∈ R^{C_out×C_in×kH×kW}, b ∈ R^{C_out}
    # Code: X [B, C_in, H_in, W_in], W [C_out, C_in, kH, kW], b [C_out]
    ### Step 1. Load from HBM: X, W, b (HBM slow)
    @staticmethod
    def forward(ctx, X, W, b, stride, padding):

        # Before nothing, get shapes
        B, C_in, H_in, W_in = X.shape

        # Normalize over C_in per spatial position — reshape to [B*H_in*W_in, C_in]
        # so that dim=-1 normalization operates over C_in exactly as in Linear
        X_flat = X.permute(0, 2, 3, 1).reshape(-1, C_in)  # [M, C_in] where M=B*H_in*W_in

        ### Step 2. On chip: Y_tilde, mu, var, r <- rms_norm_fwd(X)
        mu   = X_flat.mean(dim=-1, keepdim=True)             # [M, 1]
        var  = X_flat.var(dim=-1, keepdim=True, unbiased=False) # [M, 1]
        r    = 1.0 / (var + 1e-8).sqrt()                 # [M, 1]

        y_norm = r * (X_flat - mu)                           # [M, C_in]

        # Inside Step 3: activation_quant to F8
        s_act  = 448.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # [M, 1]
        y_tilde = (s_act * y_norm).to(torch.float8_e4m3fn).to(torch.float32) / s_act     # [M, C_in]

        # Reshape y_tilde back to [B, C_in, H_in, W_in] for conv2d
        y_tilde_4d = y_tilde.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]


        ### Step 3: On chip: W_tilde <- weight_quant(W)
        s_w     = 1.0 / W.abs().mean().clamp(min=1e-8) # scalar
        
        # w_tilde [C_out, C_in, kH, kW] -> ternary: {-mean(|W|), 0, +mean(|W|)} = {-1/s_w, 0, +1/s_w}
        # w_tilde = (s_w * W).round().clamp(-1, 1) / s_w # no gradient flows through round/clamp
        w_tilde = W - (W - (s_w * W).round().clamp(-1, 1) / s_w).detach() # STE allows gradient to flow through to W

        ### Step 4: On chip: O <- Y_tilde ⊛ W_tilde + b
        O = F.conv2d(y_tilde_4d, w_tilde, b, stride, padding)  # [B, C_out, H_out, W_out]


        ### Step 5: Store to HBM: O (return) + mu, var, r + X, W, b
        ctx.save_for_backward(X, W, b, mu, var, r)
        ctx.stride = stride
        ctx.padding = padding

        return O # [B, C_out, H_out, W_out]

    ### Step 1. Load from HBM: X, W, b, O (not used), mu, var, r, dO (HBM slow)
    @staticmethod
    def backward(ctx, dO):
        X, W, b, mu, var, r = ctx.saved_tensors
        stride, padding = ctx.stride, ctx.padding

        B, C_in, H_in, W_in = X.shape

        ### Step 2. On chip: dY <- dO × W^T
        # In conv2d, W^T means transposed convolution
        dY_4d = torch.nn.grad.conv2d_input(X.shape, W, dO.float(), stride=stride, padding=padding)  # [B, C_in, H_in, W_in]

        # Reshape dY to [M, C_in] to match RMSNorm dimension: M = B * H_in * W_in
        dY = dY_4d.permute(0, 2, 3, 1).reshape(-1, C_in)  # [M, C_in]

        ### Step 3. On chip: dX, Y_tilde <- rms_norm_bwd(dY, X, mu, var, r)
        X_flat = X.permute(0, 2, 3, 1).reshape(-1, C_in)              # [M, C_in]

        y_norm  = r * (X_flat - mu)
        
        # E4M3 (High Precision): max value: 448 /vs/ E5M2 (High Range): max value: 57,344
        s_act  = 448.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        Y_tilde = (s_act * y_norm).to(torch.float8_e4m3fn).to(torch.float32) / s_act # [M, C_in]

        dsigma2 = (dY * (X_flat - mu) * -0.5 * r**3).sum(dim=-1, keepdim=True)  # [M, 1]

        dmu = (-r * dY).sum(dim=-1, keepdim=True) + dsigma2 * (X_flat - mu).mean(dim=-1, keepdim=True)  # [M, 1]
        
        # normalization dimension is C_in
        dX_flat = r * dY + 2 * dsigma2 * (X_flat - mu) / C_in + dmu / C_in # [M, C_in]

        # Reshape dX back to [B, C_in, H_in, W_in]
        dX = dX_flat.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]


        ### Step 4: On chip: dW <- dO^T × Y_tilde
        # Reshape Y_tilde back to 4d for conv2d_weight
        Y_tilde_4d = Y_tilde.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]
        dW = torch.nn.grad.conv2d_weight(Y_tilde_4d, W.shape, dO.float(), stride=stride, padding=padding)  # [C_out, C_in, kH, kW]

        ### Step 5: On chip: db <- sum(dO)
        db = dO.sum(dim=(0, 2, 3))    # [C_out]

        ### Step 6: Store dX, dW, db to HBM
        # Same args as forward(ctx, X, W, b, stride, padding)
        return dX, dW, db, None, None   # None for stride, padding

################ MMF Conv2d Layer ################
class MMFConv2dv3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        in_channels = int(in_channels)
        out_channels = int(out_channels)
        
        # X [B, C_in, H_in, W_in] conv W [C_out, C_in, kH, kW] = O [B, C_out, H_out, W_out]

        # Weights: [C_out, C_in, kH, kW]
        self.weight  = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size)) 

        # Bias [C_out]
        self.bias    = nn.Parameter(torch.zeros(out_channels))

        # Stride and padding (scalars)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return MMFConv2dFunctionv3.apply(x, self.weight, self.bias, self.stride, self.padding)





################################ MMFv4 ################################

################ MMF Linear Function ################
class MMFLinearFunctionv4(torch.autograd.Function):
    # Theory: X ∈ R^{M×N}, W ∈ R^{N×K}, b ∈ R^K
    # Code: X [M,N], W [K, N], b [K] 
    # Step 1. Load from HBM: X, W, b (HBM slow)
    @staticmethod
    def forward(ctx, X, W, b):
        # Step 2. On chip: Y_tilde, mu, var, r <- rms_norm_fwd(X)
        mu   = X.mean(dim=-1, keepdim=True)             # [M, 1]
        var  = X.var(dim=-1, keepdim=True, unbiased=False) # [M, 1]
        r    = 1.0 / (var + 1e-8).sqrt()                 # [M, 1]

        y_norm = r * (X - mu)                           # [M, N]

        # Inside Step 2: activation_quant
        # s_act  = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # [M, 1]
        # y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act     # [M, N]

        # Step 3: On chip: W_tilde <- weight_quant(W)
        s_w     = 1.0 / W.abs().mean().clamp(min=1e-8) # scalar
        
        # w_tilde [K, N] -> ternary: {-mean(|W|), 0, +mean(|W|)} = {-1/s_w, 0, +1/s_w}
        # w_tilde = (s_w * W).round().clamp(-1, 1) / s_w # no gradient flows through round/clamp
        w_tilde = W - (W - (s_w * W).round().clamp(-1, 1) / s_w).detach() # STE allows gradient to flow through to W

        # Step 4: On chip: O <- Y_tilde ⊛ W_tilde + b
        O = y_norm @ w_tilde.t() + b    # [M, K] <- [M, N] @ [N, K] + [K]

        # Step 5: Store to HBM: O (return) + mu, var, r + X, W, b
        ctx.save_for_backward(X, W, b, mu, var, r)

        return O # [M, K]

    # Step 1. Load from HBM: X, W, b, O (not used), mu, var, r, dO (HBM slow)
    @staticmethod
    def backward(ctx, dO):
        X, W, b, mu, var, r = ctx.saved_tensors

        # Step 2. On chip: dY <- dO × W^T
        dY = dO @ W # [M, N] <- [M, K] @ [K, N] 

        # Step 3. On chip: dX, Y_tilde <- rms_norm_bwd(dY, X, mu, var, r)
        y_norm  = r * (X - mu)
        # s_act   = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        # Y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act

        dsigma2 = (dY * (X - mu) * -0.5 * r**3).sum(dim=-1, keepdim=True)  # [M, 1]

        dmu = (-r * dY).sum(dim=-1, keepdim=True) + dsigma2 * (X - mu).mean(dim=-1, keepdim=True)  # [M, 1]
        
        N  = X.shape[-1]
        dX = r * dY + 2 * dsigma2 * (X - mu) / N + dmu / N

        # Step 4: On chip: dW <- dO^T × Y_tilde
        dW = dO.t() @ y_norm   # [K, N] = [K, M] @ [M, N] 

        # Step 5: On chip: db <- sum(dO)
        db = dO.sum(dim=0)    # [K]

        # Step 6: Store dX, dW, db to HBM
        # Same args as forward(ctx, X, W, b)
        return dX, dW, db

################ MMF Linear Layer ################
class MMFLinearv4(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        in_features = int(in_features)
        out_features = int(out_features)

        # X [M,N] * W.t() [N,K] = O [M,K]
        self.weight = nn.Parameter(torch.randn(out_features, in_features)) # [K, N]
        self.bias = nn.Parameter(torch.zeros(out_features)) # [K]

    def forward(self, x):
        return MMFLinearFunctionv4.apply(x, self.weight, self.bias)

################ MMF Conv2d Function ################
class MMFConv2dFunctionv4(torch.autograd.Function):
    # Theory: X ∈ R^{B×C_in×H_in×W_in}, W ∈ R^{C_out×C_in×kH×kW}, b ∈ R^{C_out}
    # Code: X [B, C_in, H_in, W_in], W [C_out, C_in, kH, kW], b [C_out]
    ### Step 1. Load from HBM: X, W, b (HBM slow)
    @staticmethod
    def forward(ctx, X, W, b, stride, padding):

        # Before nothing, get shapes
        B, C_in, H_in, W_in = X.shape

        # Normalize over C_in per spatial position — reshape to [B*H_in*W_in, C_in]
        # so that dim=-1 normalization operates over C_in exactly as in Linear
        X_flat = X.permute(0, 2, 3, 1).reshape(-1, C_in)  # [M, C_in] where M=B*H_in*W_in

        ### Step 2. On chip: Y_tilde, mu, var, r <- rms_norm_fwd(X)
        mu   = X_flat.mean(dim=-1, keepdim=True)             # [M, 1]
        var  = X_flat.var(dim=-1, keepdim=True, unbiased=False) # [M, 1]
        r    = 1.0 / (var + 1e-8).sqrt()                 # [M, 1]

        y_norm = r * (X_flat - mu)                           # [M, C_in]

        # Inside Step 2: activation_quant
        # s_act  = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # [M, 1]
        # y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act     # [M, C_in]

        # Reshape y_tilde back to [B, C_in, H_in, W_in] for conv2d
        # y_tilde_4d = y_tilde.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]
        y_norm_4d = y_norm.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]


        ### Step 3: On chip: W_tilde <- weight_quant(W)
        s_w     = 1.0 / W.abs().mean().clamp(min=1e-8) # scalar
        
        # w_tilde [C_out, C_in, kH, kW] -> ternary: {-mean(|W|), 0, +mean(|W|)} = {-1/s_w, 0, +1/s_w}
        # w_tilde = (s_w * W).round().clamp(-1, 1) / s_w # no gradient flows through round/clamp
        w_tilde = W - (W - (s_w * W).round().clamp(-1, 1) / s_w).detach() # STE allows gradient to flow through to W

        ### Step 4: On chip: O <- Y_tilde ⊛ W_tilde + b
        O = F.conv2d(y_norm_4d, w_tilde, b, stride, padding)  # [B, C_out, H_out, W_out]


        ### Step 5: Store to HBM: O (return) + mu, var, r + X, W, b
        ctx.save_for_backward(X, W, b, mu, var, r)
        ctx.stride = stride
        ctx.padding = padding

        return O # [B, C_out, H_out, W_out]

    ### Step 1. Load from HBM: X, W, b, O (not used), mu, var, r, dO (HBM slow)
    @staticmethod
    def backward(ctx, dO):
        X, W, b, mu, var, r = ctx.saved_tensors
        stride, padding = ctx.stride, ctx.padding

        B, C_in, H_in, W_in = X.shape

        ### Step 2. On chip: dY <- dO × W^T
        # In conv2d, W^T means transposed convolution
        dY_4d = torch.nn.grad.conv2d_input(X.shape, W, dO.float(), stride=stride, padding=padding)  # [B, C_in, H_in, W_in]

        # Reshape dY to [M, C_in] to match RMSNorm dimension: M = B * H_in * W_in
        dY = dY_4d.permute(0, 2, 3, 1).reshape(-1, C_in)  # [M, C_in]

        ### Step 3. On chip: dX, Y_tilde <- rms_norm_bwd(dY, X, mu, var, r)
        X_flat = X.permute(0, 2, 3, 1).reshape(-1, C_in)              # [M, C_in]

        y_norm  = r * (X_flat - mu) # [M, C_in]
        # s_act   = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        # Y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act # [M, C_in]

        dsigma2 = (dY * (X_flat - mu) * -0.5 * r**3).sum(dim=-1, keepdim=True)  # [M, 1]

        dmu = (-r * dY).sum(dim=-1, keepdim=True) + dsigma2 * (X_flat - mu).mean(dim=-1, keepdim=True)  # [M, 1]
        
        # normalization dimension is C_in
        dX_flat = r * dY + 2 * dsigma2 * (X_flat - mu) / C_in + dmu / C_in # [M, C_in]

        # Reshape dX back to [B, C_in, H_in, W_in]
        dX = dX_flat.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]


        ### Step 4: On chip: dW <- dO^T × Y_tilde
        # Reshape Y_tilde back to 4d for conv2d_weight
        # Y_tilde_4d = Y_tilde.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]
        y_norm_4d = y_norm.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]
        dW = torch.nn.grad.conv2d_weight(y_norm_4d, W.shape, dO.float(), stride=stride, padding=padding)  # [C_out, C_in, kH, kW]

        ### Step 5: On chip: db <- sum(dO)
        db = dO.sum(dim=(0, 2, 3))    # [C_out]

        ### Step 6: Store dX, dW, db to HBM
        # Same args as forward(ctx, X, W, b, stride, padding)
        return dX, dW, db, None, None   # None for stride, padding

################ MMF Conv2d Layer ################
class MMFConv2dv4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        in_channels = int(in_channels)
        out_channels = int(out_channels)
        
        # X [B, C_in, H_in, W_in] conv W [C_out, C_in, kH, kW] = O [B, C_out, H_out, W_out]

        # Weights: [C_out, C_in, kH, kW]
        self.weight  = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size)) 

        # Bias [C_out]
        self.bias    = nn.Parameter(torch.zeros(out_channels))

        # Stride and padding (scalars)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return MMFConv2dFunctionv4.apply(x, self.weight, self.bias, self.stride, self.padding)





################################ MMFv5 ################################

################ MMF Linear Layer ################
class MMFLinearv5(nn.Module):
    def __init__(self, in_features, out_features, weight_init_scale=1):
        super().__init__()
        
        in_features = int(in_features)
        out_features = int(out_features)

        # X [M,N] * W.t() [N,K] = O [M,K]
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * weight_init_scale) # [K, N]
        self.bias = nn.Parameter(torch.randn(out_features) * weight_init_scale) # [K]

    def forward(self, x):
        return MMFLinearFunction.apply(x, self.weight, self.bias)

################ MMF Conv2d Layer ################
class MMFConv2dv5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, weight_init_scale=1):
        super().__init__()

        in_channels = int(in_channels)
        out_channels = int(out_channels)
        
        # X [B, C_in, H_in, W_in] conv W [C_out, C_in, kH, kW] = O [B, C_out, H_out, W_out]

        # Weights: [C_out, C_in, kH, kW]
        self.weight  = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * weight_init_scale)

        # Bias [C_out]
        self.bias    = nn.Parameter(torch.randn(out_channels) * weight_init_scale)

        # Stride and padding (scalars)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return MMFConv2dFunction.apply(x, self.weight, self.bias, self.stride, self.padding)




################################ MMFv6 ################################

################ MMF Linear Function ################
class MMFLinearFunctionv6(torch.autograd.Function):
    # Theory: X ∈ R^{M×N}, W ∈ R^{N×K}, b ∈ R^K
    # Code: X [M,N], W [K, N], b [K] 
    # Step 1. Load from HBM: X, W, b (HBM slow)
    @staticmethod
    def forward(ctx, X, W, b, scale_w):
        # Step 2. On chip: Y_tilde, mu, var, r <- rms_norm_fwd(X)
        mu   = X.mean(dim=-1, keepdim=True)             # [M, 1]
        var  = X.var(dim=-1, keepdim=True, unbiased=False) # [M, 1]
        r    = 1.0 / (var + 1e-8).sqrt()                 # [M, 1]

        y_norm = r * (X - mu)                           # [M, N]

        # Inside Step 2: activation_quant
        s_act  = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # [M, 1]
        y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act     # [M, N]

        # Step 3: On chip: W_tilde <- weight_quant(W)
        s_w     = scale_w / W.abs().mean().clamp(min=1e-8) # scalar
        
        # w_tilde [K, N] -> ternary: {-mean(|W|), 0, +mean(|W|)} = {-1/s_w, 0, +1/s_w}
        # w_tilde = (s_w * W).round().clamp(-1, 1) / s_w # no gradient flows through round/clamp
        w_tilde = W - (W - (s_w * W).round().clamp(-1, 1) / s_w).detach() # STE allows gradient to flow through to W

        # Step 4: On chip: O <- Y_tilde ⊛ W_tilde + b
        O = y_tilde @ w_tilde.t() + b    # [M, K] <- [M, N] @ [N, K] + [K]

        # Step 5: Store to HBM: O (return) + mu, var, r + X, W, b
        # ctx.save_for_backward(X, W, b, mu, var, r)
        ctx.save_for_backward(X, W, b, mu, var, r, w_tilde, scale_w)
        ctx.mark_non_differentiable(scale_w)

        return O # [M, K]

    # Step 1. Load from HBM: X, W, b, O (not used), mu, var, r, dO (HBM slow)
    @staticmethod
    def backward(ctx, dO):
        X, W, b, mu, var, r, w_tilde, scale_w = ctx.saved_tensors

        # Step 2. On chip: dY <- dO × W^T
        dY = dO @ W # [M, N] <- [M, K] @ [K, N] 

        # Step 3. On chip: dX, Y_tilde <- rms_norm_bwd(dY, X, mu, var, r)
        y_norm  = r * (X - mu)
        s_act   = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        Y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act

        dsigma2 = (dY * (X - mu) * -0.5 * r**3).sum(dim=-1, keepdim=True)  # [M, 1]

        dmu = (-r * dY).sum(dim=-1, keepdim=True) + dsigma2 * (X - mu).mean(dim=-1, keepdim=True)  # [M, 1]
        
        N  = X.shape[-1]
        dX = r * dY + 2 * dsigma2 * (X - mu) / N + dmu / N

        # Step 4: On chip: dW <- dO^T × Y_tilde
        dW = dO.t() @ Y_tilde   # [K, N] = [K, M] @ [M, N] 

        # Step 5: On chip: db <- sum(dO)
        db = dO.sum(dim=0)    # [K]

        # Step 6 (new): On chip: dscale_w <- compute gradient for scale_w

        # Now compute gradient through w_tilde
        dscale_w = torch.autograd.grad(
            outputs=w_tilde,           
            inputs=scale_w,
            grad_outputs=dW,
            retain_graph=False,
            allow_unused=True
        )
        
        dscale_w = dscale_w if dscale_w is not None else torch.zeros_like(scale_w)


        # Step 6: Store dX, dW, db to HBM
        # Same args as forward(ctx, X, W, b)
        return dX, dW, db, dscale_w   # None for scale_w

################ MMF Linear Layer ################
class MMFLinearv6(nn.Module):
    def __init__(self, in_features, out_features, weight_init_scale=1.0):
        super().__init__()
        
        in_features = int(in_features)
        out_features = int(out_features)

        # X [M,N] * W.t() [N,K] = O [M,K]
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * weight_init_scale) # [K, N]
        self.bias = nn.Parameter(torch.randn(out_features) * weight_init_scale) # [K]
        self.scale_w = nn.Parameter(torch.ones(1))


    def forward(self, x):
        return MMFLinearFunctionv6.apply(x, self.weight, self.bias, self.scale_w)

################ MMF Conv2d Function ################
class MMFConv2dFunction6(torch.autograd.Function):
    # Theory: X ∈ R^{B×C_in×H_in×W_in}, W ∈ R^{C_out×C_in×kH×kW}, b ∈ R^{C_out}
    # Code: X [B, C_in, H_in, W_in], W [C_out, C_in, kH, kW], b [C_out]
    ### Step 1. Load from HBM: X, W, b (HBM slow)
    @staticmethod
    def forward(ctx, X, W, b, stride, padding, scale_w):

        # Before nothing, get shapes
        B, C_in, H_in, W_in = X.shape

        # Normalize over C_in per spatial position — reshape to [B*H_in*W_in, C_in]
        # so that dim=-1 normalization operates over C_in exactly as in Linear
        X_flat = X.permute(0, 2, 3, 1).reshape(-1, C_in)  # [M, C_in] where M=B*H_in*W_in

        ### Step 2. On chip: Y_tilde, mu, var, r <- rms_norm_fwd(X)
        mu   = X_flat.mean(dim=-1, keepdim=True)             # [M, 1]
        var  = X_flat.var(dim=-1, keepdim=True, unbiased=False) # [M, 1]
        r    = 1.0 / (var + 1e-8).sqrt()                 # [M, 1]

        y_norm = r * (X_flat - mu)                           # [M, C_in]

        # Inside Step 2: activation_quant
        s_act  = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # [M, 1]
        y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act     # [M, C_in]

        # Reshape y_tilde back to [B, C_in, H_in, W_in] for conv2d
        y_tilde_4d = y_tilde.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]


        ### Step 3: On chip: W_tilde <- weight_quant(W)
        # s_w     = 1.0 / W.abs().mean().clamp(min=1e-8) # scalar
        s_w = scale_w / W.abs().mean().clamp(min=1e-8) # scalar
        
        # w_tilde [C_out, C_in, kH, kW] -> ternary: {-mean(|W|), 0, +mean(|W|)} = {-1/s_w, 0, +1/s_w}
        # w_tilde = (s_w * W).round().clamp(-1, 1) / s_w # no gradient flows through round/clamp
        w_tilde = W - (W - (s_w * W).round().clamp(-1, 1) / s_w).detach() # STE allows gradient to flow through to W

        ### Step 4: On chip: O <- Y_tilde ⊛ W_tilde + b
        O = F.conv2d(y_tilde_4d, w_tilde, b, stride, padding)  # [B, C_out, H_out, W_out]


        ### Step 5: Store to HBM: O (return) + mu, var, r + X, W, b
        ctx.save_for_backward(X, W, b, mu, var, r, w_tilde, scale_w)
        ctx.mark_non_differentiable(scale_w)
        ctx.stride = stride
        ctx.padding = padding

        return O # [B, C_out, H_out, W_out]

    ### Step 1. Load from HBM: X, W, b, O (not used), mu, var, r, dO (HBM slow)
    @staticmethod
    def backward(ctx, dO):
        X, W, b, mu, var, r, w_tilde, scale_w = ctx.saved_tensors
        stride, padding = ctx.stride, ctx.padding

        B, C_in, H_in, W_in = X.shape

        ### Step 2. On chip: dY <- dO × W^T
        # In conv2d, W^T means transposed convolution
        dY_4d = torch.nn.grad.conv2d_input(X.shape, W, dO.float(), stride=stride, padding=padding)  # [B, C_in, H_in, W_in]

        # Reshape dY to [M, C_in] to match RMSNorm dimension: M = B * H_in * W_in
        dY = dY_4d.permute(0, 2, 3, 1).reshape(-1, C_in)  # [M, C_in]

        ### Step 3. On chip: dX, Y_tilde <- rms_norm_bwd(dY, X, mu, var, r)
        X_flat = X.permute(0, 2, 3, 1).reshape(-1, C_in)              # [M, C_in]

        y_norm  = r * (X_flat - mu)
        s_act   = 127.0 / y_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        Y_tilde = (s_act * y_norm).round().clamp(-128, 127) / s_act # [M, C_in]

        dsigma2 = (dY * (X_flat - mu) * -0.5 * r**3).sum(dim=-1, keepdim=True)  # [M, 1]

        dmu = (-r * dY).sum(dim=-1, keepdim=True) + dsigma2 * (X_flat - mu).mean(dim=-1, keepdim=True)  # [M, 1]
        
        # normalization dimension is C_in
        dX_flat = r * dY + 2 * dsigma2 * (X_flat - mu) / C_in + dmu / C_in # [M, C_in]

        # Reshape dX back to [B, C_in, H_in, W_in]
        dX = dX_flat.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]


        ### Step 4: On chip: dW <- dO^T × Y_tilde
        # Reshape Y_tilde back to 4d for conv2d_weight
        Y_tilde_4d = Y_tilde.reshape(B, H_in, W_in, C_in).permute(0, 3, 1, 2)  # [B, C_in, H_in, W_in]
        dW = torch.nn.grad.conv2d_weight(Y_tilde_4d, W.shape, dO.float(), stride=stride, padding=padding)  # [C_out, C_in, kH, kW]

        ### Step 5: On chip: db <- sum(dO)
        db = dO.sum(dim=(0, 2, 3))    # [C_out]

        ### Step 6 (new): On chip: dscale_w <- compute gradient for scale_w
        
        # Now compute gradient through w_tilde
        dscale_w = torch.autograd.grad(
            outputs=w_tilde,           # ← This was the mistake
            inputs=scale_w,
            grad_outputs=dW,
            retain_graph=False,
            allow_unused=True
        )
        dscale_w = dscale_w[0] if dscale_w[0] is not None else torch.zeros_like(scale_w)

        ### Step 6: Store dX, dW, db to HBM
        # Same args as forward(ctx, X, W, b, stride, padding)
        return dX, dW, db, None, None, dscale_w   # None for stride, padding, scale_w

################ MMF Conv2d Layer ################
class MMFConv2dv6(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, weight_init_scale=1.0):
        super().__init__()

        in_channels = int(in_channels)
        out_channels = int(out_channels)
        
        # X [B, C_in, H_in, W_in] conv W [C_out, C_in, kH, kW] = O [B, C_out, H_out, W_out]

        # Weights: [C_out, C_in, kH, kW]
        self.weight  = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * weight_init_scale)

        # Bias [C_out]
        self.bias    = nn.Parameter(torch.randn(out_channels) * weight_init_scale)

        # Scalar to scale the weights before quantization
        self.scale_w = nn.Parameter(torch.ones(1)) 

        # Stride and padding (scalars)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return MMFConv2dFunction6.apply(x, self.weight, self.bias, self.stride, self.padding, self.scale_w)









################ Tests ################
if __name__ == "__main__":
    print("Testing MMFLinear...")
    x = torch.randn(32, 128, device="cuda")  # batch 32, dim 128
    lin_mmf = MMFLinearv6(128, 64).cuda()
    out_mmf = lin_mmf(x)
    print("MMF output shape:", out_mmf.shape) # [32, 64]

    print("\n") 

    print("Testing MMFConv2d...")
    x = torch.randn(16, 3, 32, 32, device="cuda", requires_grad=True)  # batch 16, 3 channels, 32×32
    conv_mmf = MMFConv2dv6(3, 64, kernel_size=3, stride=1, padding=1).cuda()
    out_mmf = conv_mmf(x)
    print("MMF output shape:", out_mmf.shape)  # [16, 64, 32, 32]

    print("\n") 

    print("Testing MMFConv2d gradients...")
    layer = MMFConv2dv6(3, 64, kernel_size=3, padding=1).cuda()
    fc = MMFLinearv6(64, 10).cuda()
    x = torch.randn(4, 3, 32, 32, device='cuda', requires_grad=True)
    labels = torch.randint(0, 10, (4,), device='cuda')
    out = layer(x)           # [4, 64, 32, 32]
    out = out.mean(dim=[2,3]) # [4, 64]
    out = fc(out)             # [4, 10]
    loss = nn.CrossEntropyLoss()(out, labels)
    loss.backward()

    print("grad norm:", layer.weight.grad.norm().item())
    print("any nan:", layer.weight.grad.isnan().any().item())
    print("grad std:", layer.weight.grad.std().item())
    print("filter 0 vs filter 1 identical:", 
        layer.weight.grad[0].equal(layer.weight.grad[1]))
    
    
    