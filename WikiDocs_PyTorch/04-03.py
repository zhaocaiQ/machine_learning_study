# 클래스로 로지스틱회귀 구현
# https://wikidocs.net/60037

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(2, 1),  # input_dim = 2, output_dim = 1
    nn.Sigmoid()  # 출력은 시그모이드 함수를 거친다
)

# 위의 코드 클래스로 구현
# 아래와 같은 클래스를 사용한 모델 구현 형식은 대부분의 파이토치 구현체에서 사용하고 있는 방식으로 반드시 숙지할 필요가 있음


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


torch.manual_seed(1)
# 훈련데이터
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# 구현한 클래스 지정
model = BinaryClassifier()

# 모델훈련
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    # H(x) 계산
    hypothesis = model(x_train)
    # cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 20 == 0:
        prediction = hypothesis >= torch.FloatTensor(
            [0.5])  # 예측값이 0.5를 넘음 True 간주
        correct_prediction = prediction.float() == y_train  # 실제값과 일치하는 경우만 True 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction)  # 정확도 계산
        print('Epoch {:4d}/{} Cost: {:6f} Accuracy {:2.2f}'.format(epoch,
              nb_epochs, cost.item(), accuracy*100))

model(x_train)
