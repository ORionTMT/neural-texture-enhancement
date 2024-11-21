import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

class ResBlock(nn.Module):
    """Basic Residual Block.
    
    Args:
        num_feat (int): Number of feature channels
        res_scale (float): Residual scale factor
    """
    def __init__(self, num_feat, res_scale=1.0):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Input tensor with shape (b, c, h, w)
            
        Returns:
            Tensor: Output tensor with shape (b, c, h, w)
        """
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out * self.res_scale

class MultiViewAttention(nn.Module):
    """Multi-View Cross Attention Module.
    
    Args:
        num_feat (int): Number of feature channels
        num_heads (int): Number of attention heads
    """
    def __init__(self, num_feat, num_heads):
        super(MultiViewAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (num_feat // num_heads) ** -0.5
        
        self.qkv = nn.Conv2d(num_feat, num_feat * 3, 1, bias=True)
        self.proj = nn.Conv2d(num_feat, num_feat, 1, bias=True)
        
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Input feature with shape (b, c, h, w)
            
        Returns:
            Tensor: Output feature with shape (b, c, h, w)
        """
        b, c, h, w = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv.unbind(1)  # Each with shape (b, num_heads, head_dim, h*w)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Aggregate
        x = (attn @ v).reshape(b, c, h, w)
        x = self.proj(x)
        return x

class DeformableAttention(nn.Module):
    """Simple Deformable Self Attention Module.
    (A simplified version for demonstration)
    
    Args:
        num_feat (int): Number of feature channels
    """
    def __init__(self, num_feat):
        super(DeformableAttention, self).__init__()
        self.offset_conv = nn.Conv2d(num_feat, 2, 3, 1, 1)
        self.attention = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Input feature with shape (b, c, h, w)
            
        Returns:
            Tensor: Output feature with shape (b, c, h, w)
        """
        # Generate offset
        offset = self.offset_conv(x)
        
        # Apply offset to grid
        b, c, h, w = x.shape
        grid = self._get_grid(h, w).to(x.device)
        offset = offset.permute(0, 2, 3, 1)
        grid = grid.unsqueeze(0) + offset
        
        # Sample features
        x = F.grid_sample(x, grid, align_corners=True)
        x = self.attention(x)
        return x
    
    def _get_grid(self, h, w):
        """Generate regular grid."""
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w)
        )
        return torch.stack((grid_x, grid_y), -1)

@ARCH_REGISTRY.register()
class TextureEnhancementNet(nn.Module):
    """Texture Enhancement Network.
    
    A neural network for enhancing texture quality.
    
    Args:
        num_in_ch (int): Number of input channels. Default: 9
        num_feat (int): Number of feature channels. Default: 64
        num_block (int): Number of residual blocks. Default: 4
        num_head (int): Number of attention heads. Default: 4
    """
    def __init__(self, num_in_ch=9, num_feat=64, num_block=4, num_head=4):
        super(TextureEnhancementNet, self).__init__()
        
        # Encoder
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # Feature transformation with residual blocks
        self.body = nn.ModuleList([ResBlock(num_feat) for _ in range(num_block)])
        
        # Multi-view attention
        self.multiview_attn = MultiViewAttention(num_feat, num_head)
        
        # Deformable attention
        self.deform_attn = DeformableAttention(num_feat)
        
        # Decoder with progressive upsampling
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)
        
        # Output convolution
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """Network forward function.
        
        Args:
            x (Tensor): Input tensor with shape (b, 9, h, w)
            
        Returns:
            Tensor: Output tensor with shape (b, 3, 4h, 4w)
        """
        # Encoder
        feat = self.relu(self.conv_first(x))
        
        # Residual blocks
        identity = feat
        for block in self.body:
            feat = block(feat)
        feat = feat + identity
        
        # Multi-view attention
        feat = self.multiview_attn(feat)
        
        # Deformable attention
        feat = self.deform_attn(feat)
        
        # Progressive upsampling
        feat = self.relu(self.pixel_shuffle1(self.upconv1(feat)))
        feat = self.relu(self.pixel_shuffle2(self.upconv2(feat)))
        
        # Output
        out = self.conv_last(feat)
        
        return out

    def init_weights(self, pretrained=None, strict=True):
        """Initialize network weights.
        
        Args:
            pretrained (str | None): Path to pretrained weights. Default: None
            strict (bool): Whether to strictly enforce that the keys match. Default: True
        """
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(state_dict, strict=strict)
        else:
            # Initialize weights normally
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)