# Prod2Vec(2015)

## Paper

[Grbovic, M., Radosavljevic, V., Djuric, N., Bhamidipati, N., Savla, J., Bhagwan, V., & Sharp, D. (2015, August). E-commerce in your inbox: Product recommendations at scale. In _Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining_ (pp. 1809-1818)**.**](https://dl.acm.org/doi/pdf/10.1145/2783258.2788627)

## PROPOSED APPROACH <a href="#proposed-approach" id="proposed-approach"></a>

제안된 방법은 이전 구매 내역(이메일로 날라온 영수증)을 이용하여 그 영수증 안의 상품들을 학습하여 추천하는 방법이다. 영수증을 패키지 상품으로, 영수증 안의 상품들 목록을 패키지의 여행 일정으로 대응시켜 하나투어 패키지 상품 추천시스템에 적용 가능할지 확인해볼 필요가 있다. Prod2vec을 발전시킨 Meta-prod2vec이 네이버 상품 추천시스템 중 유사아이템 추천시스템에 참고되었다.

제안된 방법은 상품을 저차원 공간에서의 표현(representation)으로 학습하는 방법을 제안한다. 임베딩 공간 안에서 최근접 이웃을 찾음으로써 추천이 이루어진다.

𝒮는 이메일 영수증들의 집합으로써, N명의 유저로부터 얻어진 것이다. 유저의 로그는 s=(e₁, e₂, …, e\_M)으로 구성되며 s ∈𝒮이다. 각각의 이메일 e는 Tₘ개의 상품들 p로 구성되어있음. 즉, eₘ = (pₘ₁, pₘ₂, …, pₘ\_Tₘ})이다.

**목적 : 각각의 상품 p의 D차원 표현인 vₚ를 찾는 것.** 이 때 당연하지만 유사한 아이템은 근처에 위치해야함.

<figure><img src="../.gitbook/assets/image (25).png" alt=""><figcaption><p>s는 이메일(e)의 시퀀스이며, 이메일은 product로 구성되어있다.</p></figcaption></figure>

### 저차원의 상품 임베딩 <a href="#undefined" id="undefined"></a>

#### **prod2vec** <a href="#prod2vec" id="prod2vec"></a>

prod2vec 모델은 NLP 분야에서의 용어를 빌리자면 구매 시퀀스를 문장으로, 시퀀스 안의 상품들을 단어로 보고 상품의 벡터 표현을 학습하는 것이다. 본 논문에서는 Skip-gram 방식\[24]을 사용하였다. 그리하여 아래의 목적함수를 최대화시킨다. 목적함수란 만약 사각형을 가장 크게 만들고 싶다고 가정할때 사각형의 넓이같은 것을 의미한다. MLE가 대표적인 목적함수이다.

<figure><img src="../.gitbook/assets/image (26).png" alt=""><figcaption></figcaption></figure>

같은 s 안에 있는 상품들은 임의로 배열된다. ℙ(pᵢ₊ⱼ|pᵢ)는 상품 pᵢ가 주어졌을 때 이웃하는 상품 pᵢ₊ⱼ를 관측할 확률이며 아래와 같이 소프트맥스 함수로 정의된다.

\


<figure><img src="../.gitbook/assets/image (5) (1).png" alt=""><figcaption></figcaption></figure>

**v**ₚ는 인풋, **v**ₚ'은 아웃풋 벡터 표현을 의미한다. c는 컨텍스트의 길이이다. P는 단어의 수이다.

\


<figure><img src="../.gitbook/assets/image (1) (5).png" alt=""><figcaption></figcaption></figure>

#### **bagged-prod2vec** <a href="#bagged-prod2vec" id="bagged-prod2vec"></a>

다수의 상품이 동시에 구매되었다는 정보를 고려하기 위해 skip-gram모델을 변형한 모델이다. 쇼핑백의 개념을 도입한다. 이 모델은 상품 수준이 아니라 영수증 수준에서 동작한다. 상품 벡터 표현은 아래와 같이 변형된 목적함수를 최대화함으로써 얻어진다.

<figure><img src="../.gitbook/assets/image (3) (2).png" alt=""><figcaption><p>prod2vec(수식 3.1)과의 차이는 j가 상품 수준에서 영수증 수준으로 바뀌었다는 것이다. 다른 컨텍스트의 아이템과 연산.</p></figcaption></figure>

ℙ(eₘ₊ⱼ|pₘₖ)는 이웃하고 있는 영수증 eₘ₊ⱼ를 관측할 확률이다. 영수증 eₘ₊ⱼ은 상품으로 구성되어 있으므로 eₘ₊ⱼ=(pₘ₊ⱼ,₁, …, pₘ₊ⱼ,\_Tₘ)이다. 상품 pₘₖ가 주어졌을 때 왜 한 단계 더 높은 수준인 영수증을 관측할 확률인가 헷갈릴 수도 있지만, ℙ(eₘ₊ⱼ|pₘₖ)는 다음과 같다.

<figure><img src="../.gitbook/assets/image (4) (3).png" alt=""><figcaption></figcaption></figure>

상품 구매의 시간적 정보를 반영하기 위해서 directed 언어 모델을 제안했다. 이는 컨텍스트로서 미래의 상품만 사용하겠다는 것이다\[12]. 위처럼 수정함으로써 상품 임베딩값은 미래 있을 구매 여부를 예측할 수 있도록 학습된다.

상품-to-상품 예측 모델

저차원의 상품 표현을 학습하고 난 후 다음으로 구매 할 아이템을 예측하는데 있어 몇 가지 방법이 있다.

**prod2vec-topK**\
구매한 상품이 주어지면, 모든 다른 상품들과 코사인 유사도를 계산해서 가장 유사한 top K 아이템을 추천함.

**prod2vec-cluster**\
추천의 다양성을 위해 상품들을 여러 클러스터들로 그룹핑하고, 이전에 구매한 상품이 속해있는 클러스터와 가장 연관 있는 클러스터 내의 상품을 추천한다. K-means 클러스터링을 썼으며, 상품 표현들 사이의 코사인 유사도를 기반으로 그룹핑했다. C개의 클러스터가 있다고 하자. cᵢ라는 클러스터에서 구매가 일어난 후 다음 구매는 Multinomial distribution(θᵢ₁, θᵢ₂, …, θᵢ\_C)를 따른다. θᵢⱼ는 cᵢ에서 구매가 일어난 다음 c\_j에서 구매가 일어날 확률이며 다음과 같다.\


<figure><img src="../.gitbook/assets/image (2) (3).png" alt=""><figcaption></figcaption></figure>

구매했던 상품 p가 주어졌다 → p가 어느 클러스터에 속하는지 확인 → p가 만약 cᵢ라는 클러스터에 속해있다면 cᵢ와 가장 연관된 클러스터를 여러개 찾음 → 그 속의 상품들과 p와 코사인 유사도를 계산하여 상위 K개를 추천한다.

***

**Future product만 사용**\
\[12]M. Grbovic, N. Djuric, V. Radosavljevic, and N. Bhamidipati. Search retargeting using directed query embeddings. In International World Wide Web Conference (WWW), 2015.

**Skip-gram**\
\[24]T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed representations of words and phrases and their compositionality. In NIPS, pages 3111–3119, 2013.
