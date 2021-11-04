# ResNet

이 논문이 나온 이유는 모델의 layer가 너무 깊어질수록 오히려 성능이 떨어지는 현상이 생김을 발견했기 때문이다.
이는 gradient vanishing/exploding 문제 때문에 학습이 잘 이루어지지 않기 때문이다.

위 [문제](https://velog.io/@weoqpur/%EB%82%B4%EA%B0%80-%EC%9D%B4%ED%95%B4%ED%95%9C-Batch-Normalization%EB%B0%B0%EC%B9%98-%EC%A0%95%EA%B7%9C%ED%99%94)
는 velog에 정리해 놓았다.

gradient vanishing/exploding 문제를 해결하기 위해 많은 방법이 나왔고 ResNet도 그 중 하나이다.
ResNet은 skip connection을 이용한 residual learning을 통해 layer가 깊어짐에 따른 gradient vanishing 문제를 해결하였다.

## ResNet 

기존의 neural net의 학습 목적은 input(x)을 타켓값(y)으로 mapping하는 함수 H(x)를 찾는 것이였다. 따라서 H(x)-y를 최소화하는 방향으로 학습을 진행한다.
이때 이미지 classification과 같은 문제의 경우 x에 대한 타겟값 y는 사실 x를 대변하는 것으로 y와 x의 의미가 같게끔 mapping해야한다. 즉, 강아지 사진의
pixel값이 input(x)로 주어질때 이를 2개의 label중 강아지가 1에 해당한다면 타켓값(y)를 1로 정해서 학습하는 것이 아닌 강아지 사진의 pixel값 (x)로 y를 
mapping해야한다.   
따라서 네트워크의 출력값이 x가 되도록 H(x)-x를 최소화하는 방향으로 학습을 진행한다.   
F(x) = H(x) - x를 잔차라고 하며 이 잔차를 학습하는 것은 Residual learning이라 한다.   

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbTY9tT%2FbtqBQ2AY09e%2FiyaK8IV4AWzjYvAvKK7nok%2Fimg.png)   

이때 위 그림처럼 네트워크의 output이 x가 되도록 mapping하는 것이 아닌 아래 그림처럼 마지막에 x를 더해서 네트워크의 output은 0이 되도록 mapping해서 최종
output이 x가 되도록 학습한다. 그 이유는 위 그림처럼 단순히 H(x)가 x가 되도록 residual learning으로 학습해도 결국 **gradient vanishing문제가
해결된건 아니다.** 따라서 네트워크는 0이 되도록 학습시키고 마지막에 x를 더해서 H(x)가 x가 되도록 학습하면 미분을 해도 x자체는 미분값 1을 갖기 때문에 각 layar
마다 **최소 gradient로 1은 갖도록 한 것이다.**

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fu7iAK%2FbtqBNkDoj6y%2F2Mxa3oVsS7SfoOzNZEZGU1%2Fimg.png)   
shortcut connection

따라서 layer가 아무리 깊어져도 최소 gradient로 1이상의 값을 가지므로 gradient vanishing 문제를 해결한 것이다. 정리 하자면 아래와 같다.

1. 이미지에서는 H(x) = x가 되도록 학습시킨다.
2. 네트워크의 output F(x)는 0이 되도록 학습시킨다.
3. F(x)+x=H(x)=x가 되도록 학습시키면 미분해도 F(x)+x의 미분값은 F(x) + 1로 최소 1이상이다.
4. 모든 layer에서의 gradient가 1+F'(x)이므로 gradient vanishing현상을 해결했다.


이렇게 shortcut connection으로 만든 block을 identity block이라고 한다. 그리고 ResNet은 identity block과 convolution block으로 구성되는데
각각은 아래 그림과 같다.

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcrj5v9%2FbtqBOrWkyBD%2Fyxk3PchJlnl25RRXYJ1vg0%2Fimg.png)   
identity block

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbQtwY4%2FbtqBSPHVY9d%2FXLSNe8537wDXwnrXBAjJ70%2Fimg.png)   
convolution block

단순히 identity block은 이전까지 설명했듯이 네트워크의 output F(x)에 x를 그대로 더하는 것이고 convolution block은 x역시 1x1 convolution연산
을 거친 후 F(x)에 더해주는 것이다. 그리고 ResNet은 이 두가지 block을 아래그림과 같이 쌓아서 구성한다.

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb5HQOH%2FbtqBNjqRUsk%2FYnxL1ai7kIa9peEYSRlQGK%2Fimg.png)   
ResNet structure

ResNet의 파라미터 구조는 아래 그림과 같다.   

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FupZbe%2FbtqBOrva4eX%2FiNbnXbFPj1SKFfgZsDFFvk%2Fimg.png)   

ResNet-101의 경우 각 stage마다 convolution block은 1개씩 존재한다.
따라서 위 그림을 참고하면 identity block은 각 stage에서 2,3,22,2개씩 존재하는 것이다.
