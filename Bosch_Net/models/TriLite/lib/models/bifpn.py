import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        """
        Khởi tạo FPN module.

        :param in_channels_list: Danh sách các số lượng channels của input feature maps (ví dụ: [256, 512, 1024]).
        :param out_channels: Số lượng channels của output feature maps.
        """
        super(FPN, self).__init__()
        
        # 1x1 convolutions để điều chỉnh số lượng channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # 3x3 convolutions để làm mịn các output
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
    
    def forward(self, inputs):
        """
        Xử lý forward của FPN module.

        :param inputs: Danh sách các input feature maps (ví dụ: [P3, P4, P5]).
        :return: Danh sách các output feature maps đã qua FPN.
        """
        # Step 1: Điều chỉnh số lượng channels cho mỗi input feature map
        P5, P4, P3 = inputs
        M5 = self.lateral_convs[2](P5)
        M4 = self.lateral_convs[1](P4)
        M3 = self.lateral_convs[0](P3)
        
        # Step 2: Top-down pathway
        # Up-sample M5 và cộng với P4 đã qua 1x1 conv
        M4 = M4 + F.interpolate(M5, size=M4.shape[-2:], mode='bilinear', align_corners=False)
        # Up-sample M4 và cộng với P3 đã qua 1x1 conv
        M3 = M3 + F.interpolate(M4, size=M3.shape[-2:], mode='bilinear', align_corners=False)
        
        # Step 3: Output convolutions (3x3 conv)
        O5 = self.output_convs[2](M5)
        O4 = self.output_convs[1](M4)
        O3 = self.output_convs[0](M3)
        
        return [O3, O4, O5]

# Example usage
if __name__ == "__main__":
    # Giả lập input với batch size 1
    P3 = torch.randn(1, 128, 80, 80)  # Feature map từ tầng thấp nhất
    P4 = torch.randn(1, 128, 40, 40)  # Feature map trung bình
    P5 = torch.randn(1, 128, 20, 20) # Feature map từ tầng cao nhất
    
    # Khởi tạo FPN
    fpn = FPN(in_channels_list=[128, 128, 128], out_channels=128)
    
    # Forward
    outputs = fpn([P5, P4, P3])
    
    for i, output in enumerate(outputs):
        print(f"Output {i+3} shape: {output.shape}")