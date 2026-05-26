import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)  # type: ignore
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))
def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm) # type: ignore

    return fused_conv
class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DepthwiseSeparableConv(nn.Module):
    """deepth wise separable convolution block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class AdaptiveBiFPNBlock(nn.Module):
    """Adaptive directional BiFPN module supporting upsampling/downsampling mode and outputting a single feature map."""
    def __init__(self, in_channels_list, mode='upsample'):
        """
        Args:
            in_channels_list (list): List of input feature channel sizes, e.g., [64, 128].
            out_channel (int): Unified output channel size, default is 128.
            mode (str): 'upsample' or 'downsample', determines the output resolution direction.
        """
        super().__init__()
        assert len(in_channels_list) == 2, "Two input feature layers are required."
        assert mode in ['upsample', 'downsample'], "Mode must be 'upsample' or 'downsample'."
        self.mode = mode
        out_channel = in_channels_list[0] if mode == 'upsample' else in_channels_list[1]
        # Channel adjustment convolutions
        self.conv_low = nn.Conv2d(in_channels_list[0], out_channel, 1)
        self.conv_high = nn.Conv2d(in_channels_list[1], out_channel, 1)
        
        # Learnable fusion weights (initialized to 1)
        self.weights = nn.Parameter(torch.ones(2))  # For feature fusion
        
        # 1x1 convolution before residual connection
        self.res_conv = nn.Conv2d(out_channel, out_channel, 1)
        
        # Resolution adjustment based on mode
        if self.mode == 'upsample':
            # Upsampling mode: output high-resolution features
            self.resolution_adjust = nn.Upsample(scale_factor=2)
            self.refine = nn.Sequential(
                DepthwiseSeparableConv(out_channel, out_channel),
                nn.BatchNorm2d(out_channel))
        else:
            # Downsampling mode: output low-resolution features
            self.resolution_adjust = DepthwiseSeparableConv(out_channel, out_channel, stride=2)
            self.refine = nn.Sequential(
                DepthwiseSeparableConv(out_channel, out_channel),
                nn.BatchNorm2d(out_channel))

    def forward(self, inputs):
        """
        Args:
            inputs (list): Input feature list [low_res, high_res].
                Upsampling mode: low_res=[B, C, H, W], high_res=[B, C, H/2, W/2].
                Downsampling mode: low_res=[B, C, H*2, W*2], high_res=[B, C, H, W].
        
        Returns:
            torch.Tensor: Output feature map.
        """
        low, high = inputs
        
        # Channel adjustment
        low_adjusted = self.conv_low(low)
        high_adjusted = self.conv_high(high)
        
        # Feature fusion based on mode
        if self.mode == 'upsample':
            # Upsampling mode: adjust high-resolution features to low-resolution feature size
            adjusted_feat = self.resolution_adjust(high_adjusted)
            base_feat = low_adjusted
        else:
            # Downsampling mode: adjust low-resolution features to high-resolution feature size
            adjusted_feat = self.resolution_adjust(low_adjusted)
            base_feat = high_adjusted
        
        # Learnable weight fusion
        weights = self.weights / (self.weights.sum() + 1e-6)
        fused = weights[0] * base_feat + weights[1] * adjusted_feat
        
        # Feature refinement (with residual connection)
        refined = self.refine(fused)
        
        # Residual connection (using 1x1 convolution to ensure channel matching)
        residual = self.res_conv(base_feat)
        output = refined + residual
        
        return output

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))  # k=((3,3), (3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class LargeKerAttModule(nn.Module):
    """
    Large Kernel Attention Module (LKA) for enhancing feature representation.
    This module applies depth-wise convolution, spatial convolution, and 1x1 convolution
    to generate attention maps, which are then used to modulate the input features.

    equivalent to the kernel = 31, 
    Args:
        dim (int): Number of input channels.
    Returns:
        torch.Tensor: Output feature map with the same shape as input.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  #depth-wise conv
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)   #conv_spatial
        self.conv1 = nn.Conv2d(dim, dim, 1)  # 1x1 conv

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * (attn + 1) # u * attn + u 

class C3k2LKA(nn.Module):
    """C3k2LKA is a CSP bottleneck module with large kernel attention."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """
        Initialize C3k2LKA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__()
        self.c3k = C3k(c1, c2, n, shortcut, g, e, k)
        self.lka = LargeKerAttModule(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the C3k2LKA module."""
        return self.lka(self.c3k(x))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x



class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        # assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 32) for _ in range(n))

class C2fPSALKA(nn.Module):
    """C2fPSALKA module combining C2f with Position-Sensitive Attention and Large Kernel Attention."""
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2fPSALKA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c2fpsa = C2fPSA(c1, c2, n=n, e=e)
        self.LKA = LargeKerAttModule(c2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the C2fPSALKA module.
        """
        x = self.c2fpsa(x)
        return self.LKA(x)

class UpSampleBlock(nn.Module):
    """
    UpSampleBlock class for upsampling feature maps using transposed convolution.

    This block is designed to increase the spatial resolution of input feature maps
    by a specified factor using transposed convolution.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Factor by which to upscale the input feature map.
        conv (nn.ConvTranspose2d): Transposed convolution layer for upsampling.

    Methods:
        forward: Performs a forward pass through the UpSampleBlock.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class ConcatBlock(nn.Module):
    """
    ConcatBlock class for concatenating feature maps along the channel dimension.

    This block is designed to concatenate two input feature maps along the channel dimension.

    Attributes:
        in_channel (int): Number of channels in the input feature map.

    Methods:
        forward: Performs a forward pass through the ConcatBlock.
    """

    def __init__(self, in_channel):
        super().__init__()
        self.in_channels = in_channel
        self.conv = nn.Conv2d(in_channel * 2, in_channel, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x1, x2), dim=1)
        return self.conv(x) + x1

class Backbone(nn.Module):
    """
    Backbone class for feature extraction in neural networks.

    This class is designed to extract features from input images using a series of convolutional layers,
    """
    def __init__(self, channels: List[int]):
        super().__init__()
        self.channels = channels
        assert len(self.channels) == 6, "Backbone requires 6 channel sizes for feature extraction."
        self.p1net = []
        self.p2net = []
        self.p3net = []
        self.p4net = []
        self.p5net = []
        CI, C1, C2, C3, C4, C5 = self.channels
        # p1
        self.p1net.append(Conv(CI, C1, k=3, s=2, p=1))
        # p2
        self.p2net.append(Conv(C1, C2, k=3, s=2, p=1))
        self.p2net.append(C3k2LKA(C2, C2, n=1))
        # p3
        self.p3net.append(Conv(C2, C3, k=3, s=2, p=1))
        self.p3net.append(C3k2LKA(C3, C3, n=1))
        # p4
        self.p4net.append(Conv(C3, C4, k=3, s=2, p=1))
        self.p4net.append(C3k2LKA(C4, C4, n=1))
        # p5
        self.p5net.append(Conv(C4, C5, k=3, s=2, p=1))
        self.p5net.append(C3k2LKA(C5, C5, n=1))
        self.p5net.append(SPPF(C5, C5))  # SPPF layer for p5
        self.p5net.append(C2fPSALKA(C5, C5))
        # Convert lists to nn.ModuleList for proper parameter registration
        self.p1net = nn.Sequential(*self.p1net)
        self.p2net = nn.Sequential(*self.p2net)
        self.p3net = nn.Sequential(*self.p3net)
        self.p4net = nn.Sequential(*self.p4net)
        self.p5net = nn.Sequential(*self.p5net)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.p1net(x) # type: ignore
        x2 = self.p2net(x1) # type: ignore
        x3 = self.p3net(x2) # type: ignore
        x4 = self.p4net(x3) # type: ignore
        x5 = self.p5net(x4) # type: ignore
        return x3, x4, x5  # Return the feature maps from p3, p4, and p5 layers

class Neck(nn.Module):
    """
    Neck class for feature fusion in neural networks.

    This class is designed to fuse features from different scales using Adaptive BiFPN blocks.
    """
    def __init__(self, channels: List[int]):
        super().__init__()
        self.channels = channels
        assert len(self.channels) == 3, "Neck requires 3 channel sizes for feature fusion."
        C3, C4, C5 = self.channels
        # Define the Adaptive BiFPN blocks for feature fusion
        ## upsample stage
        # First stage of BiFPN
        self.bifpn1_1 = AdaptiveBiFPNBlock([C4, C5], mode='downsample')
        self.bifpn1_2 = AdaptiveBiFPNBlock([C3, C4], mode='downsample')
        self.upsample_1 = UpSampleBlock(C5, C4, scale_factor=2)  # Upsample C5 to match C4 resolution
        self.concat_1 = ConcatBlock(C4)
        self.c3k2lka_1 = C3k2LKA(C4, C4, n=1)  # Apply C3k2LKA to C4 features
        self.upsample_2 = UpSampleBlock(C4, C3, scale_factor=2)  # Upsample C4 to match C3 resolution
        self.concat_2 = ConcatBlock(C3)
        self.c3k2lka_2 = C3k2LKA(C3, C3, n=1)  # Apply C3k2LKA to C3 features
        ## downsample stage
        # Second stage of BiFPN
        self.bifpn2_1 = AdaptiveBiFPNBlock([C3, C4], mode='upsample')        
        self.bifpn2_2 = AdaptiveBiFPNBlock([C4, C5], mode='upsample')
        self.downsample_1 = Conv(C3, C4, k=3, s=2, p=1)
        self.concat_3 = ConcatBlock(C4)
        self.c3k2lka_3 = C3k2LKA(C4, C4, n=1)
        self.downsample_2 = Conv(C4, C5, k=3, s=2, p=1)
        self.concat_4 = ConcatBlock(C5)
        self.c3k2lka_4 = C3k2LKA(C5, C5, n=1)
        

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x3, x4, x5 = x
        x5_copy = x5.clone()
        # Upsample stage
        x5 = self.bifpn1_1([x4, x5])
        x4 = self.bifpn1_2([x3, x4])
        x5 = self.upsample_1(x5)
        x4 = self.concat_1(x4, x5)
        x4 = self.c3k2lka_1(x4)
        x3 = self.concat_2(self.upsample_2(x4), x3)
        x3 = self.c3k2lka_2(x3)
        # Downsample stage
        x3 = self.bifpn2_1([x3, x4])
        x4 = self.bifpn2_2([x4, x5_copy])
        x4 = self.concat_3(x4, self.downsample_1(x3))
        x4 = self.c3k2lka_3(x4)
        x5 = self.concat_4(x5_copy, self.downsample_2(x4))
        x5 = self.c3k2lka_4(x5)
        return x3, x4, x5

class ViewAwareFeatureBlock(nn.Module):
    """
    View-specific Feature Adaptation Block (VAFB) with angle-aware positional encoding.
    Incorporates imaging view metadata into feature representations to reduce view-specific false positives.
    
    Architecture:
      1. One-hot view encoding
      2. Feature-view concatenation
      3. MLP-based feature transformation
      4. Auxiliary view classification branch
      
    """
    def __init__(self, in_channels: int = 128, hidden_dim: int = 128, num_views: int = 3):
        """
        Initialize VAFB module.
        
        Args:
            in_channels (int): Number of input feature channels (default: 128)
            num_views (int): Number of distinct view types (K)
            hidden_dim (int): Hidden dimension size for MLP (default: 256)
        """
        super().__init__()
        self.num_views = num_views
        
        # Feature transformation MLP: [features + view_encoding] -> adapted_features
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + num_views, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels)
        )
        
        # Auxiliary view classification head
        self.view_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global spatial pooling
            nn.Flatten(),
            nn.Linear(in_channels, num_views)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor, view_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with view-adaptive feature transformation.
        
        Args:
            features (torch.Tensor): Input features from BiFPN, shape [B, C, H, W]
            view_labels (torch.Tensor): View category labels, shape [B] (dtype: int64)
            
        Returns:
            tuple: 
                - fused_features: View-adapted features, shape [B, C, H, W]
                - view_loss: Auxiliary classification loss (if training), else None
        """
        B, C, H, W = features.shape
        view_labels = F.one_hot(view_labels, num_classes=self.num_views).float().to(features.device)
        features = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        view_labels = view_labels.unsqueeze(1).expand(B, H*W, self.num_views)
        fused_features = torch.cat([features, view_labels], dim=-1)
        fused_features = self.mlp(fused_features)
        fused_features = fused_features.permute(0, 2, 1).reshape(B, C, H, W)
        view_logits = self.view_classifier(fused_features)
        
        return fused_features, view_logits

def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)

class DFL(torch.nn.Module):
    # Generalized Focal Loss
    # https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=(64, 128, 256)):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box, k=3, p=1),
                                                           Conv(box, box, k=3, p=1),
                                                           torch.nn.Conv2d(box, out_channels=4 * self.ch,
                                                                           kernel_size=1)) for x in filters)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, x, k=3, p=1, g=x),
                                                           Conv(x, cls),
                                                           Conv(cls, cls, k=3, p=1, g=cls),
                                                           Conv(cls, cls),
                                                           torch.nn.Conv2d(cls, out_channels=self.nc,
                                                                           kernel_size=1)) for x in filters)
        self.vafb = torch.nn.ModuleList(ViewAwareFeatureBlock(x, x) for x in filters)

    def forward(self, x, view_labels):
        views = [0, 0, 0]
        for i, (box, cls, vafb) in enumerate(zip(self.box, self.cls, self.vafb)):
            x[i], x_logit = vafb(x[i], view_labels)
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
            views[i] = x_logit
        if self.training:
            return x, views

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1), None

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        for box, cls, s in zip(self.box, self.cls, self.stride):
            # box
            box[-1].bias.data[:] = 1.0 # type: ignore
            # cls (.01 objects, 80 classes, 640 image)
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2) # type: ignore


class YOLOv11MTB(torch.nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.net = Backbone(channels)
        c4_c6 = channels[3:]
        self.fpn = Neck(c4_c6)

        img_dummy = torch.zeros(1, channels[0], 256, 256)
        views_dummy = torch.zeros(1, dtype=torch.int64)  # Dummy view labels for initialization
        self.head = Head(num_classes, c4_c6)
        self.head.stride = torch.tensor([256 / x[0].shape[-2] for x in self.forward(img_dummy, views_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x, view_labels):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x), view_labels)

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self
