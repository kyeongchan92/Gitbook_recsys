---
description: batch size에 대한 글 여러 곳에서 발췌
---

# batch size의 의미와 학습에 미치는 영향

## 출처 : 밑바닥부터 시작하는 딥러닝

배치 처리의 이점은 컴퓨터 계산 시에 있다. 이미지 1장당 처리 시간을 대폭 줄여준다. 그 이유는 두 가지가 있다.

1. 라이브러리가 큰 배열 처리에 고도로 최적화되어 있기 때문이다.
2.  버스에 주는 부하를 줄여준다.

    배치로 처리하면 느린 I/O를 통해 데이터를 읽는 횟수가 줄고 빠른 CPU나 GPU로 순수 계산을 수행하는 비율이 높아진다.

즉, 배치로 처리하면 큰 배열 계산을 하는데, 컴퓨터에서는 작은 배열보다 큰 배열을 계산하는 것이 더 빠르다.



## 출처 : [Effect of batch size on training dynamics - Medium](https://www.google.com/url?sa=t\&rct=j\&q=\&esrc=s\&source=web\&cd=\&ved=2ahUKEwjMzLat0838AhVck1YBHdrvBLMQFnoECCcQAw\&url=https%3A%2F%2Fmedium.com%2Fmini-distill%2Feffect-of-batch-size-on-training-dynamics-21c14f7a716e\&usg=AOvVaw2Dyfvguf15ioSp4DtKPEG6)

* 큰 배치사이즈를 사용한다는 것은 모델이 매우 큰 gradient를 업데이트한다는 것을 의미한다.&#x20;
* 배치사이즈가 크면 무슨 일이 벌어지는가? --> 일반화 성능이 낮아진다. 일반화 성능이 낮다는 것은 학습데이터 오버피팅이 일어난다는 것이다. 또는, 로컬 미니멈에 빠진다.\[1]
* 배치사이즈가 작으면 무슨 일이 벌어지는가? --> gradient에 노이즈가 많이 낀다.\[2]

<figure><img src="../.gitbook/assets/image (9).png" alt=""><figcaption></figcaption></figure>

## 출처 : [Batch Size in Deep Learning](https://blog.lunit.io/2018/08/03/batch-size-in-deep-learning/)

batch size (m)을 어떻게 결정하느냐에 따라 학습 과정에 차이가 발생합니다. Batch size가 클수록 gradient가 정확해지지만 한 iteration에 대한 계산량이 늘어나게 됩니다. 그러나 한 iteration에서 각 example에 대한 gradient ![\nabla L\_i(\theta)](https://s0.wp.com/latex.php?latex=%5Cnabla+L\_i%28%5Ctheta%29\&bg=ffffff\&fg=000000\&s=0\&c=20201002)는 parellel하게 계산이 가능하므로, 큰 batch를 사용하면 multi-GPU 등 parellel computation의 활용도를 높여서 전체 학습 시간을 단축할 수 있습니다. 정리하면, batch size가 SGD에 끼치는 기본적인 영향은 아래와 같습니다.

<figure><img src="../.gitbook/assets/image (6) (3) (1).png" alt=""><figcaption></figcaption></figure>

small batch를 사용하는 것이 generalization 측면에서 더 좋은 영향을 끼친다. 그러나 아직까지 딥러닝에서 batch size의 영향에 관해 뚜렷하게 밝혀진 바는 없습니다. 최적의 성능을 얻기 위한 batch size는 모델과 task의 특성에 따라 크게 달라지며, 학습 시간을 단축하기 위한 측면에서는 batch size를 최대한 크게 키우면서 성능을 향상시키는 것이 훨씬 좋은 접근법이라고 볼 수 있습니다 \[3, 4]



\[1][How to Break GPU Memory Boundaries Even with Large Batch Sizes](https://towardsdatascience.com/how-to-break-gpu-memory-boundaries-even-with-large-batch-sizes-7a9c27a400ce)

\[2][Possible for batch size of neural network to be too small?](https://datascience.stackexchange.com/questions/52884/possible-for-batch-size-of-neural-network-to-be-too-small)

\[3] Priya Goyal et al., Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, Data@Scale, 2017

\[4] Xianyan Jia, Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes, arXiv, 2017

