# 06. 미니 배치와 데이터 로드(Mini Batch and Data Load)
# https://wikidocs.net/55580
import torch
import torch.nn as nn
import torch.nn.functional as F

# 파이토치에서는 데이터를 쉽게 다룰 수 있도록 데이터셋(Dataset)과
# 데이터로더(DataLoader)를 제공하고 이를 사용하면 미니배치학습,
# 데이터셔플, 병렬처리까지 간단히 수행가능
from torch.utils.data import TensorDataset  # 텐서데이터셋
from torch.utils.data import DataLoader  # 데이터로더

# TensorDataset은 기본적으로 텐서를 입력으로 받음
x_train = torch.FloatTensor([[73, 80, 75],
                            [93, 88, 93],
                            [89, 91, 90],
                            [96, 98, 100],
                            [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# TensorDataset의 입력으로 사용하여 dataset 저장
dataset = TensorDataset(x_train, y_train)
# 파이토치의 데이터셋을 만들었다면 데이터로더를 사용 가능
# 데이터로더는 기본적으로 2개의 인자(데이터셋, 미니배치크기)를 입력 받음
# 미니 배치의 크기는 통상적으로 2의 배수를 사용
# shuffle=True를 선택하면 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꿈
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 모델, 옵티마이저 설계
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

# 모델 훈련
nb_epochs = 20
for epoch in range(nb_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        print(batch_idx)
        print(samples)
        x_train, y_train = samples
        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:6f}'.format(
            epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()
        ))

# 예측값 출력
# 임의의 입력 [73, 80, 75]를 선언
new_var = torch.FloatTensor([[73, 80, 75]])
# 입력한 값에 대해서 예측값 y를 리턴받기
pred_y = model(new_var)
print('훈련 후 입력이 73, 80, 75일 때의 예측값:', pred_y)
