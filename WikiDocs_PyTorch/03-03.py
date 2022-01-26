# 03. 다중 선형 회귀(Multivariable Linear regression)
# https://wikidocs.net/54841
# 2.파이토치로 구현하기
# 기본라이브러리
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 랜덤시드 설정
torch.manual_seed(1)

# 훈련데이터 설정
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 가중치, 기울기 선언
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    # H(x) 계산
    hypothesis = x1_train*w1 + x2_train*w2 + x3_train*w3 + b
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))

# 위의 내용을 행렬 연산으로 구현하기
x_train = torch.FloatTensor([[73, 80, 75],
                            [93, 88, 93],
                            [89, 91, 80],
                            [96, 98, 100],
                            [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# (5x3)으로 x_train 행렬선언
print(x_train.shape)
print(y_train.shape)
# 가중치와 기울기 선언
w = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 선언
optimizer = optim.SGD([w, b], lr=1e-5)

nb_epoches = 20
for epoch in range(nb_epochs+1):
    # H(x) 계산
    # 기울기 b는 브로드캐스팅되어 각 샘플에 더해짐
    # 브로드캐스팅: 서로다른 행렬을 자동으로 크기 맞춰서 계산해주는 것
    # matmul: 행렬 곱셈 = A 마지막 차원과 B 첫번째 차원이 일치해야 함
    hypothesis = x_train.matmul(w) + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        # hypothesis.squeeze().detach():squeeze()로 차원이 1인 차원 제거 후
        # detach()는 tensor에서 이루어진 모든 연산은 기록이 되는데 detach를 하면 기록으로부터 분리됨
        # cost.item(): tensor에 저장된 값만 가져옴
        print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
            epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
        ))
