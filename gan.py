import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=12, img_channels=3, feature_g=64):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.net = nn.Sequential(
            # Input: latent_dim + num_classes
            nn.ConvTranspose2d(latent_dim + num_classes, feature_g * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 16),
            nn.ReLU(True),
            
            # (feature_g*16) x 4 x 4
            nn.ConvTranspose2d(feature_g * 16, feature_g * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            
            # (feature_g*8) x 8 x 8
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            
            # (feature_g*4) x 16 x 16
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            
            # (feature_g*2) x 32 x 32
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            
            # feature_g x 64 x 64
            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 3 x 128 x 128
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, c], 1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, num_classes=12, img_channels=3, feature_d=64):
        super(Discriminator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, img_channels * 128 * 128)
        self.img_size = 128
        
        self.net = nn.Sequential(
            # Input: img_channels*2 x 128 x 128 (image + expanded label)
            nn.Conv2d(img_channels * 2, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # feature_d x 64 x 64
            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (feature_d*2) x 32 x 32
            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (feature_d*4) x 16 x 16
            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (feature_d*8) x 8 x 8
            nn.Conv2d(feature_d * 8, feature_d * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (feature_d*16) x 4 x 4
            nn.Conv2d(feature_d * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 1 x 1 x 1
        )

    def forward(self, img, labels):
        c = self.label_emb(labels).view(img.size(0), 3, self.img_size, self.img_size)
        x = torch.cat([img, c], 1)
        return self.net(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
