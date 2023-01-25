# Node2vec(2016)

## Paper

{% embed url="https://dl.acm.org/doi/pdf/10.1145/2939672.2939754" %}

## **노드 임베딩이란?**

<figure><img src="../.gitbook/assets/image (11).png" alt=""><figcaption></figcaption></figure>

****

그래프의 각 노드가 임의의 인코더를 통과하여 임베딩 공간에 위치하는 벡터로 바뀌는 과정이다.\
그리고 이 과정은 그래프에서 유사한 노드들이 임베딩 공간에서도 근처에 있도록 맵핑하고자 하는 것이다.

&#x20;

**Node2vec의 전체적인 과정**

<figure><img src="../.gitbook/assets/image (12) (4).png" alt=""><figcaption></figcaption></figure>

크게 세 가지 단계인 전이확률 계산 단계, 랜덤워크 단계, SGD 단계로 나뉜다.

&#x20;

아래는 논문 내용

## 3. FEATURE LEARNING FRAMEWORK <a href="#3.-feature-learning-framework" id="3.-feature-learning-framework"></a>

함수 f : V →ℝᵈ는 노드를 피쳐 표현으로 매핑하는 함수라고 하자. 피쳐 표현은 벡터이며, 다운스트림 태스크를 위해 우리가 학습시켜야하는 대상이다. 한 마디로, f는 |V| × d 모양의 행렬이다.

![](<../.gitbook/assets/image (3) (5).png>)

임의의 노드 u에 대하여 Nₛ을 _network neighborhood_라고 하자. Nₛ는 이웃 샘플링 방식 S에 따라 정의된다. 우리가 목적은 아래의 목적함수를 최적화하는 것이다.

<figure><img src="../.gitbook/assets/image (8).png" alt=""><figcaption></figcaption></figure>

노드 u의 피쳐표현이 주어졌을 때 u의 이웃이 관측될 로그확률을 최대화한다. 식 (1)은 다음과 같이 쓸 수 있다.

![](<../.gitbook/assets/image (2) (4).png>)

그런데 Zᵤ는 노드 하나당 계속 다른 모든 노드들과 계산해야해서 큰 그래프에서 계산복잡도가 너무 높다. 그래서 네거티브 샘플링을 사용한다. 노드의 이웃은 딱 정해지는게 아니라, 샘플링 방식 S를 어떻게 정의하느냐에 따라 달라진다. 이제 이웃을 어떻게 정의하는지 알아보자.

### 3.1 Classic search strategies <a href="#3.1-classic-search-strategies" id="3.1-classic-search-strategies"></a>

![](<../.gitbook/assets/image (6) (3).png>)

위 그래프에서 u의 이웃을 정의해보자. 여러 방식이 있겠지만, 방식들을 동등하게 비교하기 위해 이웃의 수는 k라고 정해놓기로 하자. 대표적으로 두 가지 방식이 있다.

* BFS(넓이우선탐색) 방식은 직접 연결된 노드들만 이웃으로 사용한다.
* DFS(깊이우선탐색) 방식은 거리를 증가시켜가며 노드를 추가한다.

저자는 BFS를 여러 근거를 들어 더 효율적이라고 설명하고 있다.

&#x20;

### 3.2 node2vec <a href="#3.2-node2vec" id="3.2-node2vec"></a>

BFS와 DFS를 결합하여 유연한 샘플링 방식을 고안해본다.

#### 3.2.1 Random Walks <a href="#3.2.1-random-walks" id="3.2.1-random-walks"></a>

시작 노드와 엣지를 갖는 노드들에 대하여, 모든 엣지를 동일한 확률로 사용함.

#### 3.2.2 Search bias α <a href="#3.2.2-search-bias-a" id="3.2.2-search-bias-a"></a>



&#x20;

랜덤워크를 정의한다. 위 그림처럼 노드 t에서 v로 온 후, 현재 노드 v에 위치해있다고 하자.

**Return parameter p (안 벽)**

p(> max(q, 1))을 크게 설정하면 다시 돌아가는 길의 벽이 높아진다. 탐험을 추구하게 된다. 반면 p(< min(q, 1))를 작게 설정하면 다시 되돌아가기 쉬워지며 로컬에 머물길 좋아하게 된다.

**In-out parameter q (밖 벽)**

q > 1이면 t로 돌아가기 좋아한다(밖으로 나가는 벽이 높아진다). 시작노드로 돌아가길 좋아해서 BFS처럼 된다. 주변 노드만 추가하게 될 것이다.

반면에 q< 1이면 밖으로 나가는 벽이 낮아진다. DFS처럼 된다.

![](<../.gitbook/assets/image (2) (6).png>)

&#x20;

#### 3.2.3 The node2vec algorithm <a href="#3.2.3-the-node2vec-algorithm" id="3.2.3-the-node2vec-algorithm"></a>

![](<../.gitbook/assets/image (7) (3).png>)

&#x20;

&#x20;

&#x20;

***

참고

[3. Node Embeddings](https://velog.io/@tobigsgnn1415/Node-Embeddings#6-how-to-use-embeddings)

\
