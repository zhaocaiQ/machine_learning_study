# 05.클래스로 파이토치 모델 구현하기
# https://wikidocs.net/60036

# 1. 모델을 클래스로 구현하기
# 모델 선언 및 초기화. 단순선형회구로 input_dim=1, output_dim=1
import torch.nn.functional as F
import torch
from sre_constants import OP_IGNORE
import torch.nn as nn
model = nn.Linear(1, 1)

# 위의 모델을 클래스로 구현하기


class LinearRegressionModel(nn.Module):  # torch.nn.Module을 상속받는 파이썬 클래스
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel()

# H(x)식에 입력 x로부터 예측된 y를 얻는 것을 forward연산이라고 함
# 모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=3, output_dim=1.
model = nn.Linear(3, 1)

# 위의 모델을 클래스로 구현하기


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


model = MultivariateLinearRegressionModel()

# 2. 단순 선형 회귀 클래스로 구현하기
# 랜덤시드 설정
torch.manual_seed(1)

# 데이터 설정
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel()

# optimizer 설정/ 경사하강법= SGD/ learning rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    # H(x) 계산
    prediction = model(x_train)
    # cost 계산/파이토치에서 제공하는 평균 제곱 오차 함수사용
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        # 100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# 3. 다중 선형 회귀 클래스로 구현하기
torch.manual_seed(1)
# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


model = MultivariateLinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    # H(x) 계산
    prediction = model(x_train)
    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        # 100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
