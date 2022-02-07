# 로지스틱 회귀
# https://wikidocs.net/57805
# 시그모이드 함수
import re
from tkinter.tix import Y_REGION
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 시드 설정
torch.manual_seed(1)

# x, y값 설정
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
# 텐서로 변경
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)  # torch.Size([6, 2])
print(y_train.shape)  # torch.Size([6, 1])

# x_train을 X라 하고, 이와 곱해지는 가중치 벡터를 W라고 했을 때,
# XW가 성립되기 위해서는 W벡터의 크기가 2x1이 되야함.
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 가설 식세우기
hypothesis = 1 / (1+torch.exp(-(x_train.matmul(W)+b)))
print(hypothesis)

# 파이토치에 시그모이드 함수가 이미 있으므로 위의 식 간단히 구현 가능
hypothesis = torch.sigmoid(x_train.matmul(W)+b)
print(hypothesis)

# 현재 예측값, 실제값
print(hypothesis)
print(y_train)

# 예측값, 실제값 오차 구하기
losses = -(y_train*torch.log(hypothesis) + (1-y_train)*torch.log(1-hypothesis))
print(losses)

# 오차에 대한 평균 구하기
cost = losses.mean()
print(cost)

# 위의 비용함수의 값을 파이토치로 간단히 구현가능
losses = F.binary_cross_entropy(hypothesis, y_train)
print(losses)

# 모델훈련
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    # cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W)+b)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x)계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:6f}'.format(epoch,
              nb_epochs, cost.item()))

# 훈련 후 예측값 출력
hypothesis = torch.sigmoid(x_train.matmul(W)+b)
print(hypothesis)

# 0.5를 기준으로 이상이면 True, 이하면 False 출력
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)

# 훈련된 W와 b값 출력
print(W)
print(b)
