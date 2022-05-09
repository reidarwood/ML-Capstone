import torch
from torch import nn

class CNN(nn.Module):

    def __init__(self, input_channels=3, class_num=555):
        """
        Initializes model layers.
        :param input_channels: The number of features of each sample.
        :param class_num: The number of categories.
        """
        super().__init__()

        conv1 = [torch.nn.Conv2d(input_channels, 8, 11),
                        torch.nn.MaxPool2d(2),
                        torch.nn.SiLU()]
        conv2 = [torch.nn.Conv2d(8, 16, 9),
                        torch.nn.SiLU()]
        convs = [torch.nn.Conv2d(16, 16, 7),
                        torch.nn.Conv2d(16, 16, 5),
                        torch.nn.Conv2d(16, 16, 3)]
        sequence = conv1 + conv2
        for conv in convs:
            for _ in range(3):
                sequence.append(conv)
                sequence.append(torch.nn.SiLU())
            sequence += [torch.nn.MaxPool2d(2)]
        self.conv = nn.Sequential(*sequence)
        self.flatten = torch.nn.Flatten()
        self.l = torch.nn.Linear(256, class_num)
        # self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, X):
        """
        Applies the layers defined in __init__() to input features X.
        :param X: 4D torch tensor of shape [n, 1, 8, 8], where n is batch size.
            Represents a batch of 8 * 8 gray scale images.
        :return: 2D torch tensor of shape [n, 10], where n is batch size.
            Represents logits of different categories.
        """
        conv_out = self.flatten(self.conv(X))
        return self.l(conv_out)
