# Meta-Prod2Vec(2016)

## Paper

[Vasile, F., Smirnova, E., & Conneau, A. (2016, September). Meta-prod2vec: Product embeddings using side-information for recommendation. In Proceedings of the 10th ACM conference on recommender systems](https://arxiv.org/pdf/1607.07326.pdf)

TO-DO) 호텔데이터 돌려서 prod2vec, metaprod2vec 클러스터링 비교하기

{% embed url="https://www.kaggle.com/code/keshavramaiah/hotel-recommender/notebook" %}

## Prod2vec과의 비교

<figure><img src="../.gitbook/assets/image (1) (6).png" alt=""><figcaption></figcaption></figure>

## 3 PROPOSED APPROACH

### 3.1 Prod2Vec 복습 <a href="#prod2vec" id="prod2vec"></a>

Prod2Vec에서는 이메일에 포함된 상품들의 시퀀스에 Word2Vec을 적용하였음. Word2Vec은 같은 컨텍스트 안에 출현하는 단어들끼리는 가깝다는 ‘분포 가설(Distributional Hypothesis)’을 기반으로 한다.

그러나, Prod2Vec의 임베딩은 유저가 구매한 상품들의 시퀀스(지역적인 동시발생 정보)만 고려할 뿐, 아이템의 메타데이터같은 다른 유형의 정보를 고려하지 못한다. 구체적으로는 다음과 같은 정보를 고려하지 못한다.

* 현재 카테고리 c의 상품 p를 클릭했다면, 다음 클릭할 상품은 같은 카테고리 c 안의 p'가 될 확률이 높다.
* 현재 카테고리 c를 클릭했다면, 다음 카테고리는 c 또는 c와 가장 관련된 c'일 확률이 높다(ex. 수영복 카테고리 → 선크림 카테고리)

**Prod2Vec의 목적함수**

prod2vec 정리 글에서 아래 수식이 목적함수이고 이를 최대화 시키도록 학습된다고 하였다.\


<figure><img src="../.gitbook/assets/image (15).png" alt=""><figcaption></figcaption></figure>

이 식은 Word2vec-SkipGram의 Log-likelihood 함수와 완전히 동일하다. 이 목적함수는 아래와 같이 **가중 교차 엔트로피**(Weighted Cross Entropy) 손실을 최소화하는 최적화 문제와 같다\[23]. 즉, 식 3.1에 마이너스를 붙여 최소화해도 된다.

<figure><img src="../.gitbook/assets/image (22).png" alt=""><figcaption></figcaption></figure>

Xᵢ는 상품 i의 출현 빈도이고, Xᵢⱼᵖᵒˢ는 아이템 쌍 (i, j)가 함께 출현한 빈도이다. I는 인풋 공간이고 J는 아웃풋 공간이다. pᵢⱼ는 훈련데이터 상에서 구해지는 경험적 조건부분포(empirical conditional distribution)이다. qᵢⱼ는 모델이 예측하는 조건부분포(modeled conditional distribution)이다.

![](<../.gitbook/assets/image (3) (3).png>)

위 손실함수를 통해서 상품들의 입력 공간이 학습되어, 중앙에 위치한 단어는 주변 단어를 예측할 수 있는 능력을 갖추게 된다. 오직 히든레이어 하나와 소프트맥스 출력층 하나를 가지면서 말이다.

### 3.2 Meta-Prod2Vec <a href="#meta-prod2vec" id="meta-prod2vec"></a>

Meta-Prod2Vec은 다음과 같은 점에서 Doc2Vec과 유사하다.

* 신경망의 입력 공간과 출력 공간 두 곳 모두에서 메타 정보를 결합한다.
* 아이템과 메타데이터 사이의 인터랙션을 파라미터화 한다.

**Meta-Prod2Vec 목적함수**

Meta-Prod2Vec의 손실함수는 Prod2Vec의 손실함수에 4가지 항이 더 추가된다.

<figure><img src="../.gitbook/assets/image (8) (3).png" alt=""><figcaption></figcaption></figure>

M은 메타데이터 공간이다. 예를 들면 artist id같은 것이다.

손실함수를 하나하나 살펴보자. 상품 p의 시퀀스는 아래 그림과 같고, 모든 상품은 카테고리가 하나씩 배정되어 있다고 하자.

<figure><img src="../.gitbook/assets/image (4) (4) (1).png" alt=""><figcaption><p>데이터 예시. p는 상품, c는 카테고리</p></figcaption></figure>

&#x20;

$$L_{J|I}$$ : 일반 Skip-gram의 loss

![](<../.gitbook/assets/image (21).png>)

$$L_{I|M}$$ : $$c_i$$(중심 메타)가 출현했을 때 $$p_i$$(중심 아이템)가 출현할 확률 모델링

![](<../.gitbook/assets/image (14) (2).png>)



$$L_{J|M}$$ : $$c_i$$가 출현했을 때 $$p_{i-1}$$, $$p_{i+1}$$이 출현할 확률 모델링

![](<../.gitbook/assets/image (18).png>)



$$L_{M|I}$$ : $$p_i$$가 출현했을 때 $$c_{i-1}$$, $$c_{i+1}$$이 출현할 확률 모델링

![](<../.gitbook/assets/image (5) (5).png>)



$$L_{M|M}$$ : $$c_i$$가 출현했을 때 $$c_{i-1}$$, $$c_{i+1}$$이 출현할 확률 모델링

![](<../.gitbook/assets/image (10).png>)

위처럼 학습시킴으로써 얻을 수 있는 것은, $$p_i$$벡터는 주변 상품을 예측할 수 있을 뿐만 아니라 주변 상품의 카테고리까지 예측할 수 있는 능력이 생긴다는 것이다. 이 능력을 갖춘 벡터들은 컨텍스트로부터 학습되었다고 할 수 있다.

&#x20;

더 다양한 메타데이터 소스가 있는 경우에는 그것만의 regularization 파라미터와 함께 글로벌 loss항에 추가하면 된다.

**소프트맥스 계층**

아웃풋 공간을 아이템과 메타데이터의 아웃풋 공간을 분리해도 되고 안해도 된다. 본 논문에서는 둘을 같은 공간에 임베딩하여 normalization이 같이 진행되도록 하였다.

Word2Vec의 가장 큰 장점은 확장성(Scalability)인데, 이는 곧 Negative Sampling loss로 원래의 소프트맥스 함수를 근사함으로써 Positive 및 일부의 Negative만으로 학습할 수 있기 때문이다. 네거티브 샘플링을 위해 Loss 함수를 변형하면 아래와 같다:\
![](<../.gitbook/assets/image (20) (1).png>)

## 4. EXPERIMENT <a href="#experiment" id="experiment"></a>

* dataset : 30Music
* embedding dimension fixed to 50
* window size to 3
* the side information regularization parameter λ to 1
* epochs 10

각각의 시퀀스를 학습, 검증, 테스트 셋으로 분리한다. 시퀀스의 길이를 n이라고 할 때, 각 시퀀스의 첫 n-2 아이템을 모델에 학습시키고, n-1번째 아이템으로 성능을 측정하여 하이퍼 파라미터를 조정한다. 그리고 최종 성능을 측정할 때는 첫 n-1 아이템으로 학습시킨 후, n번째 아이템을 예측한다.

\
![](<../.gitbook/assets/image (23).png>)

학습 시퀀스의 마지막 아이템을 query item이라고 하여, 이와 가장 유사한 아이템을 추천해준다.

* Hit ratio at K (HR@K) that is equal to 1/K if the test product appears in the top K list of recommended products.
  * Hit ratio : 상위 K개의 아이템 중 라벨 아이템이 들어 있다면 1/K
* Normalized Discounted Cumulative Gain (NDCG@K) favors higher ranks of the test product in the list of recommended products.
  * 이미 알고 있는 nDCG

## Extra. code 구현 <a href="#code" id="code"></a>

\


<figure><img src="../.gitbook/assets/image (19).png" alt=""><figcaption></figcaption></figure>

## Extra. 이해를 돕기 위한 관련 글 <a href="#undefined" id="undefined"></a>

[![](https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca)Meta-Prod2Vec: Simple Product Embeddings with Side-Information](https://www.linkedin.com/pulse/meta-prod2vec-simple-product-embeddings-flavian-vasile/)

**Doc2Vec**\
\[15] Q. V. Le and T. Mikolov. Distributed representations of sentences and documents. arXiv preprint arXiv:1405.4053, 2014.

**Glove**\
\[23] J. Pennington, R. Socher, and C. D. Manning. Glove: Global vectors for word representation. In EMNLP, volume 14, pages 1532–1543, 2014.
