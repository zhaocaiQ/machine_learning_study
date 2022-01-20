#https://wikidocs.net/52460

#1차원: 벡터/ 2차원: 행렬/ 3차원: 텐서
#2. 넘파이로 텐서 만들기(벡터와 행렬 만들기)
import numpy as np

#1차원 텐서인 벡터 생성
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)
#벡터의 차원과 크기 출력
print('Rank of t:', t.ndim) #차원
print('Shape of t:', t.shape) #크기

#1-1) Numpy 기초 이해하기
#numpy에서 각 벡터의 원소 출력
print('t[0] t[1] t[-1]', t[0], t[1], t[-1])
#슬라이싱 가능
print('t[2:5] t[4:-1]', t[2:5], t[4:-1])
print('t[:2] t[:3]', t[:2], t[3:])

#2) 2D with Numpy
#2차원 행렬 생성
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)
#행렬 차원, 크기 출력
print('Rank of t:', t.ndim)
print('Shape of t:', t.shape)

#3. 파이토치 텐서 선언하기(PyTorch Tensor Allocation)
import torch

#1) 1D with PyTorch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)
#벡터 차원, 크기 출력
print(t.dim()) #Rank 차원
#Shape 크기
print(t.shape)
print(t.size())
#인덱스, 슬라이싱
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

#2) 2D with PyTorch
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])
print(t)
#차원, 크기
print(t.dim()) #차원
print(t.size()) #크기
#슬라이싱
print(t[:, 1]) #첫번째 차원 전체 선택한 후 두번째 차원의 1번 인덱스 값 출력 
print(t[:, 1].size()) #위 선택의 크기 출력
print(t[:, :-1]) # 첫번째 차원 전체 선택 후 두번째 차원에서 맨 마지막 제외 출력
print(t[:, :-1].size())

#3) 브로드캐스팅(Broadcasting)
#행렬 덧셈, 뺄셈 = 행렬 크기 일치해야 함
#행렬 곱셈 = A 마지막 차원과 B 첫번째 차원이 일치해야 함
#크기가 다른 행렬 또는 텐서에 대해 사직연산 수행할 때
#파이토치를 이용 자동으로 크기를 맞춰 연산 수행 => 브로드캐스팅

#같은크기 연산
m1 = torch.FloatTensor([[3,3]])
m2 = torch.FloatTensor([[2,2]])
print(m1+m2)
#vector + scalar
m1 = torch.FloatTensor([[1,2]])
m2 = torch.FloatTensor([3]) #[3] -> [3,3]
print(m1+m2)
#1x2 vector + 2x1 vector
m1 = torch.FloatTensor([[1,2]])
m2 = torch.FloatTensor([[3],[4]])
print(m1+m2)

# **주의**
# 브로드캐스팅은 자동으로 실행되는 기능으로 두 행렬이
# 서로 같은 행렬로 생각하여 연산을 진행 했다면 문제가 발생할 수 있고
# 오류를 찾기도 어려움

#4) 자주 사용되는 기능들
#1) 행렬 곱셈과 곱셈의 차이(Matrix Multiplication Vs. Multiplication)

#matrix multiplication 곱셈
m1 = torch.FloatTensor([[1,2],[3,4]])
m2 = torch.FloatTensor([[1],[2]])
#행렬 곱셈 = A 마지막 차원과 B 첫번째 차원이 일치해야 함
print('Shape of Matrix 1:', m1.shape) #2*2 마지막 차원 2
print('Shape of Matrix 2', m2.shape) #2*1  첫번째 차원 2
#[[1*1+2*2], [3*1+4*2]]
print(m1.matmul(m2))
#반대의 경우 오류뜸
print(m2.matmul(m1))

#multiplication 곱셈
m1 = torch.FloatTensor([[1,2],[3,4]])
m2 = torch.FloatTensor([[1],[2]])
print('Shape of Matrix 1:', m1.shape) #2*2 마지막 차원 2
print('Shape of Matrix 2', m2.shape) #2*1  첫번째 차원 2
#[[1*1, 2*1], [3*1, 4*2]]
print(m1 * m2) #2*2
print(m1.mul(m2))

#2) 평균(Mean)
#numpy에서 사용법과 매우 유사
#1차원 벡터
t = torch.FloatTensor([1,2])
print(t.mean())
#2차원 행렬
t = torch.FloatTensor([[1,2],[3,4]])
print(t.mean())
#dim 매개변수에 차원을 인자로 줌
#행렬의 첫번째 차원은 '행'을 의미: 행 제거
#[1+3/2, 2+4/2]
print(t.mean(dim=0))
#행렬의 두번째 차원은 '열'을 의미: 열 제거
#[1+2/2, 3+4/2]
print(t.mean(dim=1))
#dim=-1 마지막 차원 제거: 열의 차원 제거
print(t.mean(dim=-1))

#3) 덧셈(Sum)
#덧셈은 평균과 연산방법이나 인자 의미 동일
t = torch.FloatTensor([[1,2],[3,4]])
print(t)
print(t.sum()) #단순히 원소 전체의 덧셈
print(t.sum(dim=0)) #각 열 덧셈(행 제거)
print(t.sum(dim=1)) #각 행 덧셈(열 제거)
print(t.sum(dim=-1)) #열 제거

#4) 최대(Max)와 아그맥스(ArgMax)
#Max는 원소의 최대값 리턴, ArgMax는 최대값이 가진 인덱스 리턴
t = torch.FloatTensor([[1,2],[3,4]])
print(t)
print(t.max()) #원소값 중 최대값 리턴
#첫번째 열에서 3의 인덱스 1, 두번째 열에서 4의 인덱스 1
# => argmax 값: [1,1] 리턴
print(t.max(dim=0)) #첫번째 차원 제거/ max, argmax 값 리턴
print('Max:', t.max(dim=0)[0])
print('ArgMax:0', t.max(dim=0)[1])
print(t.max(dim=1))
print(t.max(dim=-1))
print(t.argmax()) #원소값 중 최대값의 인덱스 리턴
