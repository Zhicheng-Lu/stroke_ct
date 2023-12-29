from torch import nn
from torchvision.transforms import ToTensor
import torch
import numpy as np

class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()
        self.f_size = 16
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1), stride=(2,2)),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=2*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1), stride=(2,2)),
            nn.ReLU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(in_channels=4*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1), stride=(2,2)),
            nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1), stride=(2,2), output_padding=(1,1)),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1), stride=(2,2), output_padding=(1,1)),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4*self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1), stride=(2,2), output_padding=(1,1)),
            nn.ReLU()
        )
        self.gnn_aggr1 = nn.Sequential(
            nn.Conv2d(in_channels=self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU()
        )
        self.gnn_update1 = nn.Sequential(
            nn.Conv2d(in_channels=2*self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU()
        )
        self.dense = nn.Sequential(
            nn.Linear(in_features=2*self.f_size, out_features=2),
            nn.Softmax(dim=3)
        )
        

    def forward(self, device, patient_range, cts):
        down1 = self.down1(cts)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        up1 = self.up1(down4)
        up1_cat = torch.cat((up1, down3), 1)
        up2 = self.up2(up1_cat)
        up2_cat = torch.cat((up2, down2), 1)
        up3 = self.up3(up2_cat)
        up3 = up3[:, None, :, :, :]
        up3 = up3.repeat(1,2,1,1,1)

        for i,end in enumerate(patient_range):
            # Get start and end index for certain patient
            if i == 0:
                start = 0
            else:
                start = patient_range[i-1]

            for n1 in range(start, end):
                # _slice = up3[n1,0]
                similarities = []
                # Find similarities of all pairs
                for n2 in range(start, end):
                    if n1 == n2:
                        continue
                    similarities.append((n2, nn.functional.cosine_similarity(torch.flatten(up3[n1,0]), torch.flatten(up3[n2,0]), dim=0)))
                # Sort and slice top 5
                similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
                if len(similarities) < 3:
                    knn = [similarity[0] for similarity in similarities]
                else:
                    knn = [similarities[i][0] for i in range(len(similarities)) if i<3]
                knn = up3[knn,0]
                knn = self.gnn_aggr1(knn)
                knn = torch.mean(knn, dim=0, keepdim=False)
                up3[n1,1] = knn

        up3 = torch.reshape(up3, (up3.shape[0], up3.shape[1]*up3.shape[2], up3.shape[3], up3.shape[4]))
        gnn = self.gnn_update1(up3)
        gnn = torch.cat((gnn, down1), 1)
        # Dense
        gnn = torch.moveaxis(gnn, 1, 3)
        output = self.dense(gnn)
        output = torch.moveaxis(output, 3, 1)

        return output