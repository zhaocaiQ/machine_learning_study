#04. 파이썬 클래스(class)
#https://wikidocs.net/60034

#1. 함수(function)과 클래스(Class)의 차이
#1. 함수(function)로 덧셈기 구현하기
result = 0
def add(num):
    global result
    result += num
    return result

print(add(3))
print(add(4))

#2. 함수(function)로 두 개의 덧셈기 구현하기
result1 = 0
result2 = 0
def add1(num):
    global result1
    result1 += num
    return result1

def add2(num):
    global result2
    result2 += num
    return result2

print(add1(3))
print(add1(4))
print(add2(3))
print(add2(7))

#3. 클래스(class)로 덧셈기 구현하기
class Calculator:
    def __init__(self): #객체 생성 시 호출도리 때 실행되는 초기 함수(생성자)
        self.result = 0
    
    def add(self, num): #객체 생성 후 사용할 수 있는 함수
        self.result += num
        return self.result

#위 함수로 2개의 덧셈기 만든 것과 동일하게
#class를 이용하여 만든 덧셈기 이용가능
#코드가 훨씬 간결해짐
cal1 = Calculator()
cal2 = Calculator()

print(cal1.add(3))
print(cal1.add(4))
print(cal2.add(3))
print(cal2.add(7))