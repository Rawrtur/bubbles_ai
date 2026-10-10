import torch
from torch import nn

class HandGestureNet(nn.Module):
    def __init__(self, num_classes:int=3)->None:
        super().__init__()

        self.network: nn.Sequential = nn.Sequential(
            nn.Linear(63,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,num_classes)
        )

    def forward(self,x)->nn.Sequential:
        return self.network(x)

model: HandGestureNet = HandGestureNet(num_classes=3)

loss_fn:nn.CrossEntropyLoss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

