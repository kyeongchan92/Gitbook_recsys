---
description: Graph classification
---

# DGCNN (2018)

## Paper

[Zhang, M., Cui, Z., Neumann, M., & Chen, Y. (2018, April). An end-to-end deep learning architecture for graph classification. In _Proceedings of the AAAI conference on artificial intelligence_ (Vol. 32, No. 1).](https://ojs.aaai.org/index.php/AAAI/article/view/11782)

official code : [https://github.com/muhanzhang/pytorch\_DGCNN](https://github.com/muhanzhang/pytorch\_DGCNN)

## 1 Introduction <a href="#1-introduction" id="1-introduction"></a>

본 논문에서는 더욱 많은 노드 정보를 활용할 수 있고 글로벌 그래프 구조를 학습할 수 있는 새로운 아키텍쳐를 제안함. 본 논문이 제안하는 방식은 spatial graph convolution에 해당함.

## 2 Deep Graph Convolutional Neural Network (DGCNN) <a href="#2-deep-graph-convolutional-neural-network-dgcnn" id="2-deep-graph-convolutional-neural-network-dgcnn"></a>

![figure2](https://wikidocs.net/images/page/178490/figure2.png)

DGCNN은 다음과 같은 세 가지 단계를 순차적으로 거친다.

1. Graph convolution layers : 노드의 지엽적인 구조의 특징을 추출하고, 노드 순서를 정의한다.
2. SortPooling layer : 노드 피쳐를 사전 정의된 순서대로 정렬하고, 인풋 사이즈를 통일한다.
3. Traditional convolutional and dense layers : 정렬된 그래프 표현을 읽고 예측을 생성한다.

그래프의 인접 행렬을 A로 나타낸다. 노드의 수를 n으로 나타낸다. A는 0과 1로만 이루어져있다고 가정한다. self-loop는 없다고 가정한다. 각각의 노드는 c차원의 피쳐를 갖고있다고 하자. X∈Rn×c라는 **노드 정보 행렬**을 갖고 있다고 가정한다. X는 one-hot 인코딩이 될 수도, multi-hot 인코딩이 될 수도 있다. Γ(v)는 노드 v의 이웃 노드를 나타낸다.

### 2.1 Graph convolution layers <a href="#21-graph-convolution-layers" id="21-graph-convolution-layers"></a>

**Proposed form** A와 X가 주어지면, 본 논문에서 제안하는 graph convolutional layer는 다음 수식을 수행한다:Z=f(D\~−1A\~XW)

A\~=A+I는 셀프 루프가 더해진 인접행렬이다. D\~는 diagonal degree 행렬이다. W∈Rc×c′는 학습되는 파라미터이고, f는 비선형 활성함수이고, Z∈Rc×c′는 아웃풋 행렬이다. ![step\_one](https://wikidocs.net/images/page/178490/step\_one.png) 위 수식은 네 단계로 쪼개서 생각할 수 있다. 첫 번째, 노드 피쳐 행렬 X에 선형 변환이 적용된 형태 XW이다. 이 때 c차원에서 c'차원으로 매핑된다. 두 번째, XW 앞에 A\~를 곱한 A\~XW는 노드 정보를 자기 자신에게, 그리고 이웃 노드에게 전파하는 형태다. ![second\_step](https://wikidocs.net/images/page/178490/second\_step.png) 위 그림이 두 번째 스텝을 거친 후의 A\~XW를 보여준다. A\~는 이웃이 1이므로, A\~XW를 각 노드 관점에서 보면 이웃의 피쳐가 합쳐졌다고 볼 수 있다.

A\~XWi=∑jA\~ij(XW)i=(XW)i+∑j∈Γ(i)(XW)j

즉, i번째 행은 (XW)i 자기 자신과 i의 이웃 노드에 대한 행 (XW)j의 합이다.

![third\_step](https://wikidocs.net/images/page/178490/third\_step.png) 세 번째, 정규화하는 과정이다. 첫 번째 행, 즉 1번 노드는 3개의 피쳐가 합해졌으므로 13으로 나누어진다. 마지막 스텝은 point-wise 비선형 활성화 함수 f를 적용시킨다.

멀티스케일 substructure 피쳐를 추출하기 위하여, 위 graph convolution layer를 여러개 쌓는다.

Zt+1=f(D\~−1A\~ZtWt)

컨캣된 아웃풋 Z1:h는 각 행이 노드를 나타낸다. 이 행렬을 "feature descriptor"라고 부르기로 하자. feature descriptor의 각 행은 해당 노드의 '멀티스케일 로컬 substructure 정보'를 인코딩하고있다. 말이 참 어렵다.

**Connection with Weisfeiler-Lehman subtree kernel** 추후 작성

**Connection with propagation kernel** 추후 작성

### 2.2 The SortPooling layer <a href="#22-the-sortpooling-layer" id="22-the-sortpooling-layer"></a>

SortPooling 레이어의 주 함수는 **feature descriptor**를 정렬한다. feature descriptor의 각각은 노드를 나타내고, 전통적인 1-D 컨볼루셔널 레이어 및 밀집레이어에 입력되기 전, 일정한 순서로 정렬되어있다.

여기서 생기는 물음은 어떤 순서로 정렬해야 하는가?이다. 이미지 분류에서는 픽셀이 공간적으로 자연스럽게 정렬되어있다. 텍스트 분류에서는 사전을 이용할 수 있다. 그래프에서는 노드를 **구조적 역할(structural roles)**에 따라 정렬할 수 있다. (Niepert, Ahmed, and Kutzkov 2016)는 그래프 라벨링이라는 방법, 구체적으로는 WL라는 방법을 이용하여 전처리 시 노드를 정렬했다. 왜냐하면 최종적인 WL 색이 그래프 위상에 기반한 순서를 정의해주기 때문이다. WL에 의해 정해진 노드 순서는 그래프마다 일정하게 생성되어, 서로 다른 그래프에서 유사한 구조적 역할을 가진 노드에게는 유사한 포지션이 할당된다. 결과적으로, 신경망은 그래프 노드를 시퀀스로 바라볼 수 있고 유의미한 모델을 학습할 수 있다.

DGCNN에서도 WL color를 이용하여 노드를 정렬한다. 운이 좋게도, 그래프 컨볼루션 레이어의 아웃풋은 연속적인 WL 색깔 Zt,t=1,...,h이다. 우리는 이들을 이용해 노드를 정렬할 수 있다.

이 아이디어를 이용하여 기본적인 SortPooling 레이어를 개발했다. 이 레이어의 인풋은 n×∑1hct 형태의 Z1:h 텐서이며, 각각의 행은 노드의 feature descriptor이다. 그리고 각 칼럼은 피쳐 채널이다. SortPooling 레이어의 아웃풋은 k×∑1hct 형태의 텐서이며, k는 유저가 사전 정의한 integer값이다. 우선, Zh(마지막 레이어의 아웃풋)에 따라 SortPooling 레이어의 인풋 Z1:h를 row-wise로 정렬해보자. 우리는 이 마지막 레이어의 아웃풋을 노드의 가장 잘 정제된 연속 WL 생깔로 간주할 수 있고, 이 마지막 색깔을 이용해 노드들을 정렬할 수 있다. 이렇게 하면, 일정한 순서가 그래프 노드들에게 부여되고, 전통적인 신경망을 이 정렬된 그래프 표현에 적용할 수 있게 된다. 이상적으로는, Zh가 노드를 가능한 미세하게 다른 색깔/그룹으로 분류할 수 있도록 그래프 컨볼루션 레이어가 충분히 깊어야(h가 커야)한다.

Zh에 따른 노드 정렬은 Zh의 마지막 채널 값에 따라 내림차순으로 정렬하는 것이다. 만약 같은 값이면 바로 왼쪽 값, 즉 Zih−1, Zih−2에서의 마지막 채널값을 비교해나간다. 마치 사전에 단어가 정렬되는 것처럼, 오른쪽에서 왼쪽으로 가며 정렬한다.

정렬 후 SortPooling의 다음 함수는 아웃풋 텐서의 사이즈를 통일하는 것이다. 정렬 후에 아웃풋 텐서를 첫 번째 차원 기준으로 잘라내거나 이어붙여 n차원에서 k차원으로 만든다. 이는 그래프가 각각 다른 노드 수를 가지기 때문에 그래프 사이즈를 k통일하기 위함이다. n>k일 경우 마지막 n−k 행을 잘라내고, 반대의 경우에는 제로 행벡터를 k가 될 때까지 더한다.

graph convolution layer와 전통적인 레이어와의 연결다리 역할로서의 SortPooling은 이 레이어의 인풋의 정렬된 순서를 기억함으로써 이전 레이어의 파라미터를 사용가능하게 만들면서 그래디언트를 전달할 수 있다는 장점을 갖는다. 첨부자료에서 SortPooling의 역전파가 어떻게 가능한지 증명했다.

### 2.3 Remaining layers <a href="#23-remaining-layers" id="23-remaining-layers"></a>

SortPooling 레이어를 지난 후엔 Zsp라는 텐서를 얻게 된다. Zsp의 사이즈는 k×∑1hct이다. 각 행은 노드를 나타내며 각 열은 피쳐채널을 나타낸다. 이 텐서에 대해 CNN을 수행하기 위해, 우선 Zsp를 k(∑1hct)×1의 row-wise 벡터로 reshape한다. 그리고 난 후에 ∑1hct 크기의 필터 사이즈와 스텝을 가진 1-D 컨볼루셔널 레이어를 이용하여 노드의 피쳐 디스크립터에 대해 해당 필터를 순차적으로 수행한다. 이후 여러개의 맥스풀링 레이어 및 1-D 컨볼루셔널 레이어가 더해져 노드 시퀀스에 대한 지역적 패턴을 학습하도록 한다. 최종적으로, 완전연결 레이어 및 소프트맥스 레이어를 거치게 한다.

1-D convolutional layer는 딥러닝을 이용한 자연어처리 입문의 [이 페이지](https://wikidocs.net/80437)를 보면 이해가 쉽다.

