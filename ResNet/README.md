# ResNet

이 논문이 나온 이유는 모델의 layer가 너무 깊어질수록 오히려 성능이 떨어지는 현상이 생김을 발견했기 때문이다.
이는 gradient vanishing/exploding 문제 때문에 학습이 잘 이루어지지 않기 때문이다.

위 [문제](https://velog.io/@weoqpur/%EB%82%B4%EA%B0%80-%EC%9D%B4%ED%95%B4%ED%95%9C-Batch-Normalization%EB%B0%B0%EC%B9%98-%EC%A0%95%EA%B7%9C%ED%99%94)
는 velog에 정리해 놓았다.

gradient vanishing/exploding 문제를 해결하기 위해 많은 방법이 나왔고 ResNet도 그 중 하나이다.
ResNet은 skip connection을 이용한 residual learning을 통해 layer가 깊어짐에 따른 gradient vanishing 문제를 해결하였다.

