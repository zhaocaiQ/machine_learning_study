# 07. 커스텀 데이터셋(Custom Dataset)
# https://wikidocs.net/57165

# torch.utils.data.Dataset은 파이토치에서 데이터셋을 제공하는 추상 클래스
# 이런 torch.utils.data.Dataset을 상속받아 직접 커스텀 데이터셋(Custom Dataset)을 만듦
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    # 데이터셋의 전처리를 해주는 부분
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    # 데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return len(self.x_data)

    # 데이터셋에서 특정 1개의 샘플을 가져오는 함수
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data)
        y = torch.FloatTensor(self.y_data)
        return x, y


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#모델, 옵티마이저설계
model = torch.nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

# 모델 훈련
nb_epochs = 20
for epoch in range(nb_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        # H(x)계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch: {}/{} Cost:{:.6f}'.format(
            epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()
        ))

# 임의의 입력 [73, 80, 75]를 선언
new_var = [[73, 80, 75]]
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)
