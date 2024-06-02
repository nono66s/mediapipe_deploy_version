import torch

class LSTM (torch.nn.Module):
    def __init__(self,label_name_list ,input_size=36, hidden_size=512 , num_layers=16):
        
        '''
        模型参数

        input_size   # 每个时间步的输入特征数
        hidden_size  # 隐状态特征数
        num_layers   # LSTM 层数
        output_size  # 输出特征数 #label_name_list

        '''

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层：LSTM层处理输入序列，并输出每个时间步的隐藏状态。我们选择最后一个时间步的隐藏状态作为特征向量。
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.num_classes = len(label_name_list)

        # 定义LSTM层


        # 删除Flatten层：直接使用LSTM的输出，而不是将其展开。

        # 定义分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size, 512),  # Dense
            torch.nn.ReLU(inplace=True),

            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 512),  # Dense
            torch.nn.ReLU(inplace=True),

            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, self.num_classes), # Dense，输出根据类别数量修改

            #torch.nn.Softmax(dim=1) # 在输出层不需要显式地添加Softmax，因为我们会在损失函数中处理这个操作。
        )


    def forward(self, x):

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 LSTM
        out_x, _ = self.lstm(x, (h0, c0))
        # x, _ = self.lstm(x, None)

        # 获取最后一个时间步的输出
        out_x = out_x[:, -1, :]

        # 传入分类器
        x = self.classifier(out_x)

        return x




# class LSTM (torch.nn.Module):
#     def __init__(self,label_name_list ,input_size=10, hidden_size=20 , num_layers=10):
        
#         '''
#         模型参数

#         input_size   # 每个时间步的输入特征数
#         hidden_size  # 隐状态特征数
#         num_layers   # LSTM 层数
#         output_size  # 输出特征数 #label_name_list

#         '''

#         super().__init__()

#         self.hidden_size = hidden_size
#         self.num_layers = num_layers


#         self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

#         self.fc = torch.nn.Linear(in_features=hidden_size, out_features=len(label_name_list)) 


#     def forward(self, x):
#         # # 初始化隐藏状态和细胞状态
#         # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
#         # # 前向传播 LSTM

#         # out_x, out_y = self.lstm(x, (h0, c0))
        
#         out_x, out_y = self.lstm(x, None)

#         # 解码最后一个时间步的隐状态
#         out_x = self.fc(out_x[:, -1, :])
        
#         return out_x
    

# 详细解释
# 定义 LSTM 模型：我们定义了一个名为 LSTMModel 的类，继承自 nn.Module。在这个类中，我们初始化了一个 LSTM 层和一个全连接层。

# 设置模型参数：定义 LSTM 的输入大小、隐藏层大小、层数以及输出大小。这些参数决定了 LSTM 的架构。

# 创建模型实例：使用定义的参数实例化我们的 LSTM 模型。

# 创建输入数据：输入数据的形状为 (batch_size, seq_length, input_size)。在这个例子中，我们随机生成了一个形状为 (5, 7, 10) 的输入张量。

# 前向传播：将输入数据传递给模型，得到输出。输出的形状应为 (batch_size, output_size)，在这个例子中为 (5, 1)。

# 注意事项
# 初始化隐藏状态：每次输入一个新的序列时，我们都需要初始化隐藏状态和细胞状态。这些状态通常初始化为零。
# LSTM 输出：LSTM 的输出包含两个部分：out 和 (h_n, c_n)。out 是所有时间步的隐状态，h_n 和 c_n 分别是最后一个时间步的隐状态和细胞状态。在很多情况下，我们只需要使用 out。
# 批处理：在使用批处理时，确保输入数据的形状正确，即 (batch_size, seq_length, input_size)，并且在前向传播时也要相应地调整隐藏状态和细胞状态的形状。
# 通过上述步骤，你可以在 PyTorch 中创建并使用 LSTM 模型来处理序列数据。