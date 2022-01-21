#03. 텐서 조작하기(Tensor Manipulation) 2
#https://wikidocs.net/52846

#4) 뷰(View): 원소의 수 유지하면서 크기 변경
#뷰 = 넘파이 리쉐이프(Reshape): 텐서의 크기 변경해 줌
import numpy as np
import torch
t = np.array([[[0,1,2],
             [3,4,5]],
             [[6,7,8],
             [9,10,11]]])
ft = torch.FloatTensor(t)
print(ft.shape)

#4-1) 3차원 텐서에서 2차원 텐서로 변경(차원변경)
#3차원(2,2,3) -> 2차원(2*2,3)
#원소개수 동일(2*2*3 =12) == (4*3 =12)
print(ft.view([-1,3])) #ft텐서 (?,3)dml 크기로 변경
print(ft.view([-1,3]).shape)

#4-2) 3차원 텐서의 크기 변경(크기변경)
#원소개수 동일(2*2*3 =12) == (4*1*3 =12)
print(ft.view([-1,1,3]))
print(ft.view([-1,1,3]).shape)

#5) 스퀴즈(Squeeze): 차원이 1인 경우 해당 차원 제거
ft = torch.FloatTensor([[0],[1],[2]])
print(ft)
print(ft.shape)
#(3,1) -> (3,)
print(ft.squeeze())
print(ft.squeeze().shape)

#6) 언스퀴즈(Unsqueeze): 특정 위치에 1인 차원을 추가한다.
ft = torch.FloatTensor([0,1,2])
print(ft.shape)
#첫번째 차원에 1인 차원 추가
print(ft.unsqueeze(0)) #인덱스 0 = 첫번째 차원
print(ft.unsqueeze(0).shape)
#view로 동일하게 구현 가능
print(ft.view(1,-1))
print(ft.view(1,-1).shape)

#두번째 차원에 1인 차원 추가
print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)
#현재 ft텐서는 (3,)크기이기 때문에 1과 -1 결과 같음
print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)

#7) 타입 캐스팅(Type Casting)
#32비트, 64비트/CPU, GPU 자료형
#LongTensor: Data type(64-bit integer)
lt = torch.LongTensor([1,2,3,4])
print(lt)
print(lt.float())
#Byte타입
bt = torch.ByteTensor([True, False, False, True])
print(bt)
#long타입, float타입 텐서변경
print(bt.long())
print(bt.float())

#8) 두 텐서 연결하기(concatenate)
#(2, 2)크기 동일한 텐서
x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[5,6],[7,8]])
#dim을 통해 어느 차원 늘릴지 설정가능
print(torch.cat([x,y], dim=0)) #첫번째 차원 (2,2) -> (4,2)
print(torch.cat([x,y], dim=1)) #두번째 차원 (2,2) -> (2,4)

#9) 스택킹(Stacking): 텐서 연결하는 또다른 방법
#스택킹에는 많은 연산을 포함 => 편리
x = torch.FloatTensor([1,4])
y = torch.FloatTensor([2,5])
z = torch.FloatTensor([3,6])
#스택킹으로 x,y,z 텐서 쌓기
print(torch.stack([x,y,z])) #(2,) -> (3,2)
#cat으로 위와 동일 코드 만들기
#unsqueeze로 (2,) -> (1,2)로 변경 후 cat/dim=0으로 (1,2) -> (3,2)
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))

#스택킹도 dim 인자 가능
print(torch.stack([x,y,z], dim=1)) #(2,) -> (2,3)

#10) ones_like와 zeros_like - 0으로 채워진 텐서와 1로 채워진 텐서
x = torch.FloatTensor([[0,1,2],[2,1,0]])
print(x)
print(torch.ones_like(x))  #x와 동일한 크기이면서 값은 1로 채워짐
print(torch.zeros_like(x)) # "" 위와 동일 값만 0으로 채워짐

#11) In-place Operation (덮어쓰기 연산)
x = torch.FloatTensor([[1,2],[3,4]])
print(x.mul(2.))  #각 값에 곱하기 2/ 값 x는 변경되지 않음
print(x.mul_(2.)) #위와 동일 but 값 x 변경됨
print(x)
