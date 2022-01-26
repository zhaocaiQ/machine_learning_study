# 04. nn.Module로 구현하는 선형 회귀
# https://wikidocs.net/55409
# 이전 챕터에서는 가설, 손실함수를 직접 정의함
# 이번에는 파이토치에서 이미 구현되어져 제공되고 있는 함수들을 불러오는 것으로 더 쉽게 선형 회귀 모델을 구현

# 1. 단순 선형 회귀 구현하기

# 필요 라이브러리
from sre_constants import OP_IGNORE
import torch
import torch.nn as nn
import torch.nn.functional as F
# 랜덤시드설정
torch.manual_seed(1)

# y=2x라는 식을 가정한 데이터 정의
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 선형회귀 모델구현
# 모델 선언 및 초기화,
# 단순 선형회귀로 input_dim, output_dim은 각 1로 설정
# ->하나의 입력 x에 대해 하나의 출력 y를 가지므로 입출력 차원 모두 1로 설정
model = nn.Linear(1, 1)

# model에 저장되어 있는 가중치, 기울기 출력
print(list(model.parameters()))
# [Parameter containing:
# tensor([[0.5153]], requires_grad=True), Parameter containing:
# tensor([-0.4414], requires_grad=True)]
# 위의 출력된 값은 모두 현재는 랜덤 초기화 상태,
# 두 값 모두 학습 대상으로 requires_grad=True로 설정되어 있음

# optimizer 정의
# optimizer 설정. 경사하강법 SGD 사용. learning rate는 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 전체 훈련 데이터에 대한 경사하강법 2000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):
    # H(x) 계산/가설세우기를 nn.Linear()로 쉽게 가능
    #이전: hypothesis = x1_train*w1 + x2_train*w2 + x3_train*w3 + b
    prediction = model(x_train)  # forward연산

    # cost 계산/파이토치에서 제공하는 평균 제곱 오차 함수사용
    #이전: cost = torch.mean((hypothesis - y_train) ** 2)
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 손실함수를 미분하여 gradient 계산
    # backward연산: backward() 호출하면 해당 수식의 w에 대한 기울기 계산
    # 비용 함수로부터 기울기를 구하라는 의미
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        # 100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# 훈련 잘되었는지 확인
# x에 임의의 값 4 넣어 모델이 예측하는 y값 확인
new_var = torch.FloatTensor([[4.0]])
# forward연산: H(x)식에 입력 x로부터 예측된 y를 얻는 것을 forward 연산이라고 함
pred_y = model(new_var)
# y = 2x 이므로 x값 입력이 4라면 y는 8에 가까운 값이 나와야 함.
print('훈련 후 입력이 4일 때의 예측값:', pred_y)
# 훈련 후 입력이 4일 때의 예측값: tensor([[7.9989]], grad_fn=<AddmmBackward0>)
# 7.9989로 8에 가까운 값 예측함.

# 훈련된 가중치와 기울기 출력
print(list(model.parameters()))
# [Parameter containing:
# tensor([[1.9994]], requires_grad=True), Parameter containing:
# tensor([0.0014], requires_grad=True)]
# W는 1.99, b는 0.0014로 W값이 2에 가깝고 b의 값이 0에 가깝게 훈련될 것을 알 수 있음

# 2. 다중 선형 회귀 구현하기
# 필요 라이브러리
# 랜덤시드 설정
torch.manual_seed(1)
# H(x) = w1*x1 + w2*x2 + w3*x3 + b
# 3개의 x로부터 y 예측
# 데이터 정의
x_train = torch.FloatTensor([[73, 80, 75],
                            [93, 88, 93],
                            [89, 91, 90],
                            [96, 98, 100],
                            [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# 선형회귀모델 구현/x 3개, y하나로 입출력 차원 => (3,1)
model = nn.Linear(3, 1)

# 가중치, 기울기 출력
print(list(model.parameters()))

# opimizer 설정
# model.parameters()로 가중치, 기울기 전달
# 학습률은 0.00001(1e-5)로 설정: 0.01로 하지않는 이유는 기울기가 발산하기 때문
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    # H(x) 계산
    prediction = model(x_train)
    # model(x_train) = model.forward(x_train) 동일

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# x에 임의의 값 넣어 y값 예측
new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)
print('훈련 후 입력이 73,80,75일 때 예측값:', pred_y)
# 위의 데이터는 훈련에 쓰인 데이터로 원래 y값 152임
# 아래와 같이 151.23으로 예측이 잘되는 것을 볼 수 있음
# 훈련 후 입력이 73,80,75일 때 예측값: tensor([[151.2306]], grad_fn=<AddmmBackward0>)

# 훈련된 가중치, 기울기 출력
print(list(model.parameters()))
