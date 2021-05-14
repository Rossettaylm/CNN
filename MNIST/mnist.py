import numpy as np
import torch as t

#  0. prepare, 读取数据


def read_labels(filename, items):
    with open(filename, 'rb') as file_label:
        # 除去8字节的文件头，共有items个标记的数据,一个数据占据一个字节
        file_label.seek(8)
        data = file_label.read(items)
        y = np.zeros(items, dtype=np.int64)
        for i in range(items):
            y[i] = data[i]

    return y


y_train = read_labels(filename='train-labels-idx1-ubyte', items=60000)
y_test = read_labels(filename='t10k-labels-idx1-ubyte', items=10000)


def read_images(filename, items):
    with open(filename, 'rb') as file_image:
        file_image.seek(16)
        data = file_image.read(items*28*28)
        x = np.zeros(items*28*28, dtype=np.float32)
        for i in range(items*28*28):
            x[i] = data[i] / 255

        x = x.reshape(-1, 28 * 28)
        return x


X_train = read_images(filename='train-images-idx3-ubyte', items=60000)
X_test = read_images(filename='t10k-images-idx3-ubyte', items=10000)

# 设置参数
num_epochs = 100       # 训练轮数
learning_rate = 1e-3    # 学习率
batch_size = 64         # 每批量大小


#  1.构建模型
class TestNet(t.nn.Module):
    #  初始化传入输入、隐藏层、输出3个参数
    def __init__(self, in_dim, hidden, out_dim):
        super(TestNet, self).__init__()
        # 全连接层
        self.layer1 = t.nn.Sequential(
            t.nn.Linear(in_dim, hidden), t.nn.ReLU(True))
        # 输出层
        self.layer2 = t.nn.Linear(hidden, out_dim)

    # 传入计算值的函数， 真正的计算部分
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# 输入28 × 28, 隐藏层121, 输出10类
model = TestNet(28*28, 121, 10)


#  2. 编译部分
optimizer = t.optim.SGD(model.parameters(), lr=learning_rate)


# 3. 训练部分
X_train_size = len(X_train)     # 训练集大小
for epoch in range(num_epochs):
    print('Epoch: ', epoch)     # 打印轮次
    X = t.autograd.Variable(t.from_numpy(X_train))      # 保存训练参数
    y = t.autograd.Variable(t.from_numpy(y_train))

    i = 0
    while i < X_train_size:
        # 取一个新批次的数据
        X0 = X[i: i + batch_size]
        y0 = y[i: i + batch_size]
        i += batch_size

        # 正向传播
        # 用神经网络计算10类输出结果
        out = model(X0)

        # 计算神经网络结果和实际标签结果的差值
        loss = t.nn.CrossEntropyLoss()(out, y0)

        # 反向梯度下降
        # 清空梯度
        optimizer.zero_grad()

        # 根据误差函数求导
        loss.backward()
        # 进行一轮梯度下降计算
        optimizer.step()
    print('The loss of train epoch: ', loss.item())        # type: ignore 打印损失值


# 4. 验证部分
# 将模型设计为验证模式
model.eval()
X_val = t.autograd.Variable(t.from_numpy(X_test))
y_val = t.autograd.Variable(t.from_numpy(y_test))

# 用训练好的模型验证结果
out_val = model(X_val)
loss_val = t.nn.CrossEntropyLoss()(out_val, y_val)

# 打印出测试损失值
print('The loss of test: ', loss_val.item())

# 求出最大元素的位置
_, pred = t.max(out_val, 1)

# 将预测值和标注值进行对比
num_correct = (pred == y_val).sum()

# 打印正确率
print('The accuracy: {0:2.2%}'.format(num_correct.data.numpy()/len(y_test)))
