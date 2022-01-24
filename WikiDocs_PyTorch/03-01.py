#4. 파이토치로 선형 회귀 구현하기
#https://wikidocs.net/53560

#1.기본 라이브러리
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#동일한 결과얻기 위한 시드(Seed) 설정
torch.manual_seed(1)
#2.훈련데이터 선언
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

#훈련데이터 크기(shape)출력
print(x_train.shape)
print(y_train.shape)

#3.가중치와 편햔의 초기화
#선형회귀: 학습데이터와 잘 맞는 하나의 직선 찾는 일

#가중치 W를 0으로 초기화, 학습통해 값이 변경되는 변수임 명시
#requires_grad: True로 주변 이 변수는 학습을 통해 계속 값이 변경되는 변수임을 의미
W = torch.zeros(1, requires_grad=True)
print(W) #가중치 초기화로 0이 출력됨
#편향 b도 동일하게 명시
b = torch.zeros(1, requires_grad=True)
print(b)

#4. 가설 세우기
hypothesis = x_train * W + b
print(hypothesis)

#5. 비용 함수 선언하기
#torch.mean으로 평균 구하기
cost = torch.mean((hypothesis - y_train) ** 2)
print(cost)

#6. 경사 하강법 구현하기
#SGD: 경사하강법 일종
#lr: 학습률(learning rate)
optimizer = optim.SGD([W,b], lr=0.01)
#기울기 0으로 초기화
#->새로운 가중치 편향에 대해 새로운 기울기 구할 수 있음
optimizer.zero_grad()
#가중치 W, 편향 b에 대한 기울기 계산
cost.backward()
#경사하강법 최적화 함수 optimizer의 .step()함수를 호출
#-> 인수로 들어갔던 W와 b에서 리턴되는 변수들의 기울기에
#   학습률(learning rate) 0.01을 곱하여 빼줌
optimizer.step()

#7. 전체코드
#데이터
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])
#모델초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
#optimizer 설정
optimizer = optim.SGD([W,b], lr=0.01)

nb_epochs= 1999 #원하는만큼 경사하강법 반복
for epoch in range(nb_epochs+1):
    #H(x) 계산
    hypothesis = x_train*W + b

    #cost계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    #cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    #100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))

# Epoch 1900/1999 W: 1.996659, b: 0.008 Cost: 0.000008
#훈련결과 최적의 기울기 W는 2에 가깝고, b는 0에 가까운 것을 볼 수 있음
#훈련데이터가 1,2,3 -> 2,4,6인 것을 감안하면
#실제 정답은 W가 2이고, b가 0인 H(x) = 2x이므로 거의 정답과 가까움

#optimizer.zero_grad()가 필요한 이유
#파이토치는 미분을 통해 얻은 기울기를 이전의 기울기 값에 누적시킴
import torch
w = torch.tensor(2.0, requires_grad=True)

nb_epochs =20
for epoch in range(nb_epochs+1):
    z = 2*w

    z.backward()
    print('수식을 w로 미분한 값 : {}'.format(w.grad))
#수식을 w로 미분한 값 : 2.0
#...
#수식을 w로 미분한 값 : 42.0
#=> optimier.zero_grad()를 통해 미분값을 계속 0으로 초기화 시켜줘야 함

#6.torch.manual_seed()를 하는 이유
#torch.manual_seed()를 사용한 프로그램의 결과는 다른 컴퓨터에서도 동일한 결과를 얻음
import torch
torch.manual_seed(3)
print('랜덤 시드가 3일 때')
#2개의 랜덤 수 출력
for i in range(1,3):
    print(torch.rand(1))

#랜덤 시드 변경
torch.manual_seed(5)
print('랜덤 시드가 5일 때')
for i in range(1,3):
    print(torch.rand(1))

#다시 랜덤 시드 3
torch.manual_seed(3)
print('랜덤 시드가 3일 때')
for i in range(1,3):
    print(torch.rand(1))

# ** 텐서에는 requires_grad라는 속성이 있는데 True로 하면 자동 미분이 적용 됨
#위의 속성이 적용된 텐서에 연산을 하면, 계산 그래프가 생성되며
#backward 함수를 호출하면 그래프로부터 자동으로 미분계산됨