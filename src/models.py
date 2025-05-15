import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class BoundingBoxAdjustmentModel(nn.Module):
    def __init__(
        self,
        image_height,
        image_width,
        resnet_out_channels=2048,
        feature_map_downsample_ratio=32,
    ):
        super(BoundingBoxAdjustmentModel, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        # feature_map_downsample_ratio は ResNet からの特徴マップの解像度を計算するのに使うが、
        # F.interpolate で直接出力サイズを指定するため、直接的な影響は少ない。
        # しかし、概念として残しておく。

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_features = nn.Sequential(
            *list(resnet.children())[:-2]
        )  # Remove avgpool and fc

        # --- Vertical Pooling and Decoder (for Horizontal Edges: Top, Bottom) ---
        # Vertical Pooling: (B, C, fH, fW) -> (B, C, fH) by averaging over fW
        # 1D Decoder for Top edge: input (B, C, fH), output (B, 1, fH) -> (B, image_height)
        self.decoder_top = nn.Sequential(
            nn.Conv1d(
                resnet_out_channels, resnet_out_channels // 4, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                resnet_out_channels // 4, 1, kernel_size=1
            ),  # Output: (B, 1, fH_actual)
        )
        # 1D Decoder for Bottom edge
        self.decoder_bottom = nn.Sequential(
            nn.Conv1d(
                resnet_out_channels, resnet_out_channels // 4, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                resnet_out_channels // 4, 1, kernel_size=1
            ),  # Output: (B, 1, fH_actual)
        )

        # --- Horizontal Pooling and Decoder (for Vertical Edges: Left, Right) ---
        # Horizontal Pooling: (B, C, fH, fW) -> (B, C, fW) by averaging over fH
        # 1D Decoder for Left edge: input (B, C, fW), output (B, 1, fW) -> (B, image_width)
        self.decoder_left = nn.Sequential(
            nn.Conv1d(
                resnet_out_channels, resnet_out_channels // 4, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                resnet_out_channels // 4, 1, kernel_size=1
            ),  # Output: (B, 1, fW_actual)
        )
        # 1D Decoder for Right edge
        self.decoder_right = nn.Sequential(
            nn.Conv1d(
                resnet_out_channels, resnet_out_channels // 4, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                resnet_out_channels // 4, 1, kernel_size=1
            ),  # Output: (B, 1, fW_actual)
        )

    def forward(self, image_tensor):
        # image_tensor is expected to be (B, 3, H_model_input, W_model_input)
        # e.g., (B, 3, CROP_IMG_HEIGHT, CROP_IMG_WIDTH)

        features = self.resnet_features(
            image_tensor
        )  # (B, C_resnet, fH_actual, fW_actual)
        # fH_actual, fW_actual are determined by H_model_input, W_model_input and ResNet architecture

        # Vertical Pooling (Average over width dimension)
        # (B, C_resnet, fH_actual, fW_actual) -> (B, C_resnet, fH_actual)
        pooled_vertical = torch.mean(features, dim=3)

        # Horizontal Edges (Top, Bottom)
        # self.decoder_top/bottom expects input (B, C_resnet, fH_actual)
        pred_top_logits_fH = self.decoder_top(pooled_vertical).squeeze(
            1
        )  # (B, fH_actual)
        pred_bottom_logits_fH = self.decoder_bottom(pooled_vertical).squeeze(
            1
        )  # (B, fH_actual)

        # Upsample to self.image_height (which is CROP_IMG_HEIGHT)
        # Input to interpolate: (B, Channels_in=1, Length_in=fH_actual)
        # Output: (B, Channels_out=1, Length_out=self.image_height)
        pred_top_logits = F.interpolate(
            pred_top_logits_fH.unsqueeze(1),
            size=self.image_height,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        pred_bottom_logits = F.interpolate(
            pred_bottom_logits_fH.unsqueeze(1),
            size=self.image_height,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

        # Horizontal Pooling (Average over height dimension)
        # (B, C_resnet, fH_actual, fW_actual) -> (B, C_resnet, fW_actual)
        pooled_horizontal = torch.mean(features, dim=2)

        # Vertical Edges (Left, Right)
        # self.decoder_left/right expects input (B, C_resnet, fW_actual)
        pred_left_logits_fW = self.decoder_left(pooled_horizontal).squeeze(
            1
        )  # (B, fW_actual)
        pred_right_logits_fW = self.decoder_right(pooled_horizontal).squeeze(
            1
        )  # (B, fW_actual)

        # Upsample to self.image_width (which is CROP_IMG_WIDTH)
        pred_left_logits = F.interpolate(
            pred_left_logits_fW.unsqueeze(1),
            size=self.image_width,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        pred_right_logits = F.interpolate(
            pred_right_logits_fW.unsqueeze(1),
            size=self.image_width,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

        # Return logits; sigmoid will be applied in loss function (BCEWithLogitsLoss) or during inference
        return {
            "top": pred_top_logits,
            "bottom": pred_bottom_logits,
            "left": pred_left_logits,
            "right": pred_right_logits,
        }
