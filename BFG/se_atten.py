import torch
from torch import nn
from torchstat import stat  


class se_block(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(se_block, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):  

        b, c, h, w = inputs.shape
        x = self.avg_pool(inputs)
        x = x.view([b, c])

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        x = x.view([b, c, 1, 1])

        outputs = x * inputs
        return outputs


if __name__ == "__main__":
    inputs = torch.rand(4, 32, 16, 16)
    in_channel = inputs.shape[1]
    model = se_block(in_channel=in_channel)

    outputs = model(inputs)
    print(outputs.shape)  # [4,32,16,16])

    print(model)  
    stat(model, input_size=[32, 16, 16])  
