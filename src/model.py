import torch
import torch.nn as nn
from torchvision import models

class LandmarkModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Khởi tạo backbone MobileNetV2 pretrained
        self.backbone = models.mobilenet_v2(pretrained=True)

        # 2. Xử lý đầu vào 1 kênh (Grayscale: 112x112x1) thay vì RGB
        old_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        # Giữ lại trọng số pretrained bằng cách cộng các kênh màu lại
        new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True)
        self.backbone.features[0][0] = new_conv

        # 3. Thay thế nn.ReLU6 bằng nn.ReLU tối ưu cho phần cứng (FPGA)
        def replace_relu6_with_relu(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU6):
                    setattr(module, name, nn.ReLU(inplace=getattr(child, 'inplace', False)))
                else:
                    replace_relu6_with_relu(child)
        replace_relu6_with_relu(self.backbone)

        # 4. Thêm lớp Conv-BatchNorm-ReLU 3x3 trước Regression Head (trước AdaptivePooling)
        # MobileNetV2's features là nn.Sequential nên ta có thể add_module
        self.backbone.features.add_module('extra_layer', nn.Sequential(
            nn.Conv2d(1280, 1280, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True)
        ))

        # 5. Cập nhật Classifier (Regression Head)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 196) # 98 điểm Landmark (x, y)
        )

    def forward(self, x):
        # backbone() của mobilenet_v2 sẽ chạy: features -> adaptive_avg_pool2d -> flatten -> classifier
        return self.backbone(x)