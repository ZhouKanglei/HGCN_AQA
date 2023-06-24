import torch.nn as nn
import torch
from models.i3d import I3D


class I3D_backbone(nn.Module):
    def __init__(self, I3D_ckpt_path, I3D_class=400):
        super(I3D_backbone, self).__init__()
        print('\tUsing I3D backbone,', end=' ')
        # self.backbone = InceptionI3d()
        self.backbone = I3D(I3D_class)
        self.load_pretrain(I3D_ckpt_path)

    def load_pretrain(self, I3D_ckpt_path):
        self.backbone.load_state_dict(torch.load(I3D_ckpt_path))
        print('loading ckpt done.')

    def get_feature_dim(self):
        return self.backbone.get_logits_dim()

    def forward(self, video):
        batch_size, C, frames, H, W = video.shape

        # spatial-temporal feature
        if frames >= 160:
            start_idx = [i for i in range(0, frames, 16)]
        else:
            start_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 87]

        clips = [video[:, :, i:i + 16] for i in start_idx]

        clip_feats = torch.empty(batch_size, 1024, 10).to(video.device)

        for i in range(len(start_idx)):
            clip_feats[:, :, i] = self.backbone(clips[i]).reshape(batch_size, 1024)

        return clip_feats


if __name__ == '__main__':
    x = torch.randn((8, 3, 103, 224, 224))

    model = I3D_backbone(I3D_ckpt_path='../weights/model_rgb.pth', I3D_class=400)

    y = model(x)
