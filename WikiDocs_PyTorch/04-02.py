# nn.Module로 로지스틱회귀 구현
# https://wikidocs.net/58686

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
# 훈련데이터 설정
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# nn.Sequential()은 nn.Module 층을 차례로 쌓을 수 있도록 함
model = nn.Sequential(
    nn.Linear(2, 1),  # input_dim = 2, output_dim = 1
    nn.Sigmoid()  # 출력은 시그모이드 함수를 거침
)

model(x_train)

# 모델훈련
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = model(x_train)

    # cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor(
            [0.5])  # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train  # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction)  # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(  # 각 에포크마다 정확도를 출력
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))

model(x_train)

print(list(model.parameters()))
