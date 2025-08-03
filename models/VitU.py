import torch
import torch.nn as nn
from timm.models.vision_transformer import vit_base_patch16_224
from einops import rearrange


class ViTEncoder(nn.Module):
    def __init__(self, image_size=256, patch_size=16, in_channels=3, embed_dim=768, freeze=True):
        super().__init__()
        self.vit = vit_base_patch16_224(pretrained=True)
        self.vit.patch_embed.img_size = (224, 224)
        self.proj = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.embed_dim = embed_dim

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.eval()
            
    def forward(self, x):
        # print(f"[DEBUG] Input x shape BEFORE projection: {x.shape}")

        x_proj = self.proj(x)
        
        B = x_proj.shape[0]

        x_patch = self.vit.patch_embed(x_proj)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x_tokens = torch.cat((cls_tokens, x_patch), dim=1)
        x_tokens = x_tokens + self.vit.pos_embed
        x_tokens = self.vit.pos_drop(x_tokens)
        x_tokens = self.vit.blocks(x_tokens)
        x_tokens = self.vit.norm(x_tokens)
        x_tokens = x_tokens[:, 1:]

        N = x_tokens.shape[1]
        grid_size = int(N ** 0.5)
        assert grid_size * grid_size == N, f"Expected square grid, got N={N}"
        vit_feats = rearrange(x_tokens, 'b (h w) c -> b c h w', h=grid_size, w=grid_size)
        return vit_feats, x_proj


class DecoderUNetDeep(nn.Module):
    def __init__(self, in_channels=768, out_channels=3, skip=True):
        super().__init__()
        self.skip = skip

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14→28
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28→56
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56→112
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 112→224
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.outconv = nn.Conv2d(64 + (3 if skip else 0), out_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x, x_skip=None):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        if self.skip and x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        return self.tanh(self.outconv(x))



class ViTUNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, image_size=256, patch_size=16, freeze_encoder=True, use_skip=True):
        super().__init__()
        self.encoder = ViTEncoder(image_size=image_size, patch_size=patch_size,
                                  in_channels=input_nc, embed_dim=768, freeze=freeze_encoder)
        self.decoder = DecoderUNetDeep(in_channels=768, out_channels=output_nc, skip=use_skip)

    def forward(self, x):
        feats, x_skip = self.encoder(x)
        out = self.decoder(feats, x_skip if self.decoder.skip else None)
        return out
