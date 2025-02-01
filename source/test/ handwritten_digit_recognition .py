import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as torch_optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])    # 定义数据预处理：将图像转换为张量
    data_set = datasets.MNIST("", is_train, transform=to_tensor, download=True)    # 加载 MNIST 数据集
    return DataLoader(data_set, batch_size=15, shuffle=True)   # 创建 DataLoader 对象

def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():  # 禁用梯度计算
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28*28))            # 将输入数据展平为 (batch_size, 28*28)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:                # 检查预测是否正确
                    n_correct += 1
                n_total += 1
    return n_correct / n_total    # 返回准确率


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(28*28, 64)  # 全连接层
        self.fc2 = nn.Linear(64, 64)  # 全连接层
        self.fc3 = nn.Linear(64, 64)  # 全连接层
        self.fc4 = nn.Linear(64, 10)   # 全连接层

    def forward(self, x):#x表示图像
        # 定义前向传播
        x = torch.nn.functional.relu(self.fc1(x))  # 第一层全连接 + ReLU激活
        x = torch.nn.functional.relu(self.fc2(x))  # 第二层全连接 + ReLU激活
        x = torch.nn.functional.relu(self.fc3(x))  # 第二层全连接 + ReLU激活
        x = torch.nn.functional.log_softmax(self.fc4(x),dim=1)
        return x
    

def main():
    train_data = get_data_loader(is_train=True)#训练集
    test_data = get_data_loader(is_train=False)#测试集
    net = Net()

    print("Initial accuracy:", evaluate(test_data, net))
    optimizer = torch_optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(4):
        for (x, y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1, 28*28))#正向传播
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()#反向误差传播
            optimizer.step()
        print("Epoch", epoch, "accuracy:", evaluate(test_data, net))
    # 假设 test_data 是一个 DataLoader 对象，net 是一个训练好的神经网络模型
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        # 获取模型预测结果
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        # 创建图像窗口
        plt.figure(n)
        # 显示图像
        plt.imshow(x[0].view(28, 28), cmap='gray')
        # 设置标题为预测结果
        plt.title("prediction: " + str(int(predict)))
    # 显示所有图像
    plt.show()

if __name__ == "__main__":
    main()