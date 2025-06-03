import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights


class BoundingBoxAdjustmentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Split ResNet layers for multi-scale feature extraction
        self.resnet_conv1 = resnet.conv1
        self.resnet_bn1 = resnet.bn1
        self.resnet_relu = resnet.relu
        self.resnet_maxpool = resnet.maxpool

        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4

        # Feature dimensions from ResNet50
        self.features_dim_l3 = 1024  # Output channels of layer3
        self.features_dim_l4 = 2048  # Output channels of layer4

        # Decoders for layer3 features
        self.decoder_l3_top = nn.Linear(self.features_dim_l3, self.config.IMAGE_SIZE)
        self.decoder_l3_bottom = nn.Linear(self.features_dim_l3, self.config.IMAGE_SIZE)
        self.decoder_l3_left = nn.Linear(self.features_dim_l3, self.config.IMAGE_SIZE)
        self.decoder_l3_right = nn.Linear(self.features_dim_l3, self.config.IMAGE_SIZE)

        # Decoders for layer4 features
        # (These are equivalent to the original single-scale decoders if num_resnet_features was 2048)
        self.decoder_l4_top = nn.Linear(self.features_dim_l4, self.config.IMAGE_SIZE)
        self.decoder_l4_bottom = nn.Linear(self.features_dim_l4, self.config.IMAGE_SIZE)
        self.decoder_l4_left = nn.Linear(self.features_dim_l4, self.config.IMAGE_SIZE)
        self.decoder_l4_right = nn.Linear(self.features_dim_l4, self.config.IMAGE_SIZE)

    def forward(self, image_crop):  # Input is the cropped image region
        # Pass through ResNet backbone
        x = self.resnet_conv1(image_crop)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)

        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)
        features_l3 = self.resnet_layer3(
            x
        )  # Features from layer3: (B, 1024, H/16, W/16)
        features_l4 = self.resnet_layer4(
            features_l3
        )  # Features from layer4: (B, 2048, H/32, W/32)

        # --- Process features from layer3 ---
        # Directional Spatial Pooling
        # Vertical pooling for top/bottom edges: (B, C, H, W) -> (B, C, W_feat)
        pooled_v_l3 = torch.mean(features_l3, dim=2)
        # Horizontal pooling for left/right edges: (B, C, H, W) -> (B, C, H_feat)
        pooled_h_l3 = torch.mean(features_l3, dim=3)

        # 1D-Decoders for layer3
        # Input to Linear: (B, W_feat, C) or (B, H_feat, C)
        # Output from Linear: (B, W_feat, IMAGE_SIZE) or (B, H_feat, IMAGE_SIZE)
        preds_top_l3_raw = self.decoder_l3_top(pooled_v_l3.permute(0, 2, 1))
        preds_bottom_l3_raw = self.decoder_l3_bottom(pooled_v_l3.permute(0, 2, 1))
        preds_left_l3_raw = self.decoder_l3_left(pooled_h_l3.permute(0, 2, 1))
        preds_right_l3_raw = self.decoder_l3_right(pooled_h_l3.permute(0, 2, 1))

        # Average predictions over the feature map's spatial dimension
        # Output shape: (B, IMAGE_SIZE)
        preds_top_l3 = torch.mean(preds_top_l3_raw, dim=1)
        preds_bottom_l3 = torch.mean(preds_bottom_l3_raw, dim=1)
        preds_left_l3 = torch.mean(preds_left_l3_raw, dim=1)
        preds_right_l3 = torch.mean(preds_right_l3_raw, dim=1)

        # --- Process features from layer4 ---
        # Directional Spatial Pooling
        pooled_v_l4 = torch.mean(features_l4, dim=2)  # (B, C, W_feat)
        pooled_h_l4 = torch.mean(features_l4, dim=3)  # (B, C, H_feat)

        # 1D-Decoders for layer4
        preds_top_l4_raw = self.decoder_l4_top(pooled_v_l4.permute(0, 2, 1))
        preds_bottom_l4_raw = self.decoder_l4_bottom(pooled_v_l4.permute(0, 2, 1))
        preds_left_l4_raw = self.decoder_l4_left(pooled_h_l4.permute(0, 2, 1))
        preds_right_l4_raw = self.decoder_l4_right(pooled_h_l4.permute(0, 2, 1))

        # Average predictions
        preds_top_l4 = torch.mean(preds_top_l4_raw, dim=1)
        preds_bottom_l4 = torch.mean(preds_bottom_l4_raw, dim=1)
        preds_left_l4 = torch.mean(preds_left_l4_raw, dim=1)
        preds_right_l4 = torch.mean(preds_right_l4_raw, dim=1)

        # --- Combine predictions from layer3 and layer4 (simple averaging) ---
        # Sigmoid will be applied by BCEWithLogitsLoss during training
        final_preds_top = (preds_top_l3 + preds_top_l4) / 2.0
        final_preds_bottom = (preds_bottom_l3 + preds_bottom_l4) / 2.0
        final_preds_left = (preds_left_l3 + preds_left_l4) / 2.0
        final_preds_right = (preds_right_l3 + preds_right_l4) / 2.0

        return final_preds_top, final_preds_bottom, final_preds_left, final_preds_right


# Example usage (for testing model structure, not part of the class):
if __name__ == "__main__":
    # Mock config object
    class MockConfig:
        IMAGE_SIZE = 224

    mock_config = MockConfig()

    model = BoundingBoxAdjustmentModel(config=mock_config)
    model.eval()  # Set to evaluation mode

    # Create a dummy input tensor (batch_size, channels, height, width)
    # This represents the cropped image region fed to the model
    dummy_image_crop = torch.randn(2, 3, mock_config.IMAGE_SIZE, mock_config.IMAGE_SIZE)

    with torch.no_grad():  # No need to track gradients for this test
        preds_top, preds_bottom, preds_left, preds_right = model(dummy_image_crop)

    print("Output shapes:")
    print(f"Top: {preds_top.shape}")
    print(f"Bottom: {preds_bottom.shape}")
    print(f"Left: {preds_left.shape}")
    print(f"Right: {preds_right.shape}")

    # Expected output shape for each: (batch_size, IMAGE_SIZE)
    # e.g., torch.Size([2, 224]) for batch_size=2 and IMAGE_SIZE=224
    assert preds_top.shape == (2, mock_config.IMAGE_SIZE)
    assert preds_bottom.shape == (2, mock_config.IMAGE_SIZE)
    assert preds_left.shape == (2, mock_config.IMAGE_SIZE)
    assert preds_right.shape == (2, mock_config.IMAGE_SIZE)
    print("\\nModel structure seems okay for multi-scale processing.")
