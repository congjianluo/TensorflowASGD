import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.autograd import Variable

LR = 0.01
BATCH_SIZE = 40
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))


def get_batch_data(step):
    return x[step * BATCH_SIZE:(step + 1) * BATCH_SIZE], \
           y[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]


plt.scatter(x.numpy(), y.numpy())
plt.show()

torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


new_SGD = Net()
new_ASGD = Net()
nets = [new_SGD, new_ASGD]

opt_SGD = torch.optim.SGD(new_SGD.parameters(), lr=LR)
opt_ASGD = torch.optim.ASGD(new_ASGD.parameters(), lr=LR, lambd=0, alpha=0)
optimizers = [opt_SGD, opt_ASGD]

loss_function = torch.nn.MSELoss()
losses_his = [[], []]

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        for net, opt, l_his in zip(nets, optimizers, losses_his):
            output = net(b_x)
            loss = loss_function(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data[0])

labels = ["SGD", "ASGD"]
for i, l_his in enumerate(losses_his):
    print(l_his)
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel("Steps")
plt.ylabel("Loss")
# plt.ylim(0, 0.2)
plt.show()
# optimizer = torch.optim.SGD()
#
# torch.optim.
