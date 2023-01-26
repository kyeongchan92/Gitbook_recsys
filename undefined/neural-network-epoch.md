# Neural Network에서의 Epoch

[Epoch in Neural Networks](https://www.baeldung.com/cs/epoch-neural-networks)를 번역한 글입니다.

### **1. Introduction** <a href="#1-introduction" id="1-introduction"></a>

신경망에서 epoch이라는 것이 어떤 의미를 갖는지 알아보자. 그리고 neural network의 학습 수렴 과정과 에폭 수의 관계도 알아보자. 그래서 더욱 일반화가 잘 된 모델을 어떻게 early stopping을 통해 얻을 수 있는지도 알아보자.

### **2. Neural Networks** <a href="#2-neural-networks" id="2-neural-networks"></a>

Neural network는 **지도(supervised)** 기계학습 알고리즘이다. Neural network를 이용하면 분류 문제나 회귀 문제를 해결할 수 있다. 그러나 neural network 사용에 있어서 장점과 단점이 있다.

neural network를 구축하는 것은 구조 지향적인(architecture-oriented) 질문에 답하는 것과 같다. 복잡한 문제와 그에 관련되어 사용 가능한 데이터를 가지고서, neural network를 학습시키는데 있어서 사이즈와 깊이는 모두 다르다. 게다가, 인풋 정보를 전처리하고, 학습가중치를 초기화하고, 편향을 추가하고, 적당한 활성함수를 고르는 과정도 필요하다.

### **3. Epoch in Neural Networks** <a href="#3-epoch-in-neural-networks" id="3-epoch-in-neural-networks"></a>

epoch은 neural network를 전체 학습 데이터에 한 번 학습시키는 것을 의미한다. 한 번의 에폭에서 그 데이터를 단 한 번 사용한다. 순전파, 역전파 또한 한 번 카운팅된다.

![forward\_backward](https://wikidocs.net/images/page/180544/forward\_backward.png)

1 에폭은 하나 또는 하나 이상의 **배치**로 구성된다. 즉, neural network를 학습시킬 때 데이터셋의 일부만 사용한다. 이 일부 데이터셋을 배치라고 하고, 배치 데이터를 전파하는 것을 iteration이라고 부른다.

에폭과 iteration을 혼동해서 쓰기도 한다. 명확히 하기 위해 다음과 같은 간단한 예시를 들어보자. 1,000개의 데이터가 있다고 하자.

&#x20;

<figure><img src="https://wikidocs.net/images/page/180544/batch.png" alt=""><figcaption></figcaption></figure>

만약 배치사이즈가 1,000이라면, 1 iteration에 1에폭을 완료할 수 있다. 만약 배치사이즈가 500이라면, 2 iteration이 필요하다. 만약 배치사이즈가 100이라면, 10 iteration만에 1에폭을 완료할 수 있다. 즉, data size // batch size + 1만큼의 iteration이 필요하다(a//b : a/b의 몫).

학습할 때, 우리는 여러 번의 에폭을 사용한다. 이 말인 즉슨 neural network에 같은 데이터가 여러 번(에폭 수 만큼) 학습된다는 것이다.

### **4. Neural Network Training Convergence** <a href="#4-neural-network-training-convergence" id="4-neural-network-training-convergence"></a>

Neural network의 구조를 결정짓는 것은 모델 구축에 있어서 큰 작업이다. 그럼에도 불구하고, 우리는 모델을 훈련시키고 도중에 더 많은 하이퍼 파라미터를 조정해야 한다.

학습 과정 동안, 우리는 모델이 새로운 데이터에 잘 일반화되도록 할 뿐만 아니라, error rate를 최소화하는 것을 목표로 한다. 편향-분산 트레이드오프는 다른 지도학습 머신 러닝 알고리즘에서와 마찬가지로 여전히 우리가 피하고 싶은 함정이다.
