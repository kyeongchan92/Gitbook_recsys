# batch size의 의미와 학습에 미치는 영향

## 밑바닥에서 시작하는 딥러닝

배치 처리의 이점은 컴퓨터 계산 시에 있다. 이미지 1장당 처리 시간을 대폭 줄여준다. 그 이유는 두 가지가 있다.

1. 라이브러리가 큰 배열 처리에 고도로 최적화되어 있기 때문이다.
2.  버스에 주는 부하를 줄여준다.

    배치로 처리하면 느린 I/O를 통해 데이터를 읽는 횟수가 줄고 빠른 CPU나 GPU로 순수 계산을 수행하는 비율이 높아진다.

즉, 배치로 처리하면 큰 배열 계산을 하는데, 컴퓨터에서는 작은 배열보다 큰 배열을 계산하는 것이 더 빠르다.



## [Effect of batch size on training dynamics - Medium](https://www.google.com/url?sa=t\&rct=j\&q=\&esrc=s\&source=web\&cd=\&ved=2ahUKEwjMzLat0838AhVck1YBHdrvBLMQFnoECCcQAw\&url=https%3A%2F%2Fmedium.com%2Fmini-distill%2Feffect-of-batch-size-on-training-dynamics-21c14f7a716e\&usg=AOvVaw2Dyfvguf15ioSp4DtKPEG6)

* 큰 배치사이즈를 사용한다는 것은 모델이 매우 큰 gradient를 업데이트한다는 것을 의미한다.&#x20;
* 배치사이즈가 크면 무슨 일이 벌어지는가? --> 일반화 성능이 낮아진다. 일반화 성능이 낮다는 것은 학습데이터 오버피팅이 일어난다는 것이다. 또는, 로컬 미니멈에 빠진다.\[1]
* 배치사이즈가 작으면 무슨 일이 벌어지는가? --> gradient에 노이즈가 많이 낀다.\[2]

<figure><img src="../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

\[1][How to Break GPU Memory Boundaries Even with Large Batch Sizes](https://towardsdatascience.com/how-to-break-gpu-memory-boundaries-even-with-large-batch-sizes-7a9c27a400ce)

\[2][Possible for batch size of neural network to be too small?](https://datascience.stackexchange.com/questions/52884/possible-for-batch-size-of-neural-network-to-be-too-small)

