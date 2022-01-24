#02.자동미분(Autograd)
#https://wikidocs.net/60754
#경사하강법코드에 requires_grad=True, backward() 등이 나옴
#-> 파이토치에서 제공하는 자동미분 기능 수행

#1.경사 하강법
#경사 하강법은 비용함수(손실함수)를 미분하여 기울기를 구해
#비용(손실)이 최소화 되는 방향을 찾아내는 알고리즘
#자동미분을 통해 경사 하강법을 손쉽게 사용 가능

#2.자동미분(Autograd) 실습
#2w**2 + 5라는 식을 세우고 w에 대해 미분
import torch
#값이 2인 임의의 스칼라 텐서 w를 선언
#requires_grad=True: 이 텐서에 대한 기울기 저장(w.grad에 w에 대한 미분값 저장됨)
w = torch.tensor(2.0, requires_grad=True)
#수식 정의
y = w**2
z = 2*y + 5

#해당 수식을 w에 대해서 미분
#.backward() 호출하면 해당 수식의 w에 대한 기울기 계산
z.backward()
print('수식을 w로 미분한 값 : {}'.format(w.grad))
