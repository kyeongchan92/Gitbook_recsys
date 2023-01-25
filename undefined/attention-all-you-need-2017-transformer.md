# Attention all you need(2017, Transformer 제안) 복습

## Paper

[Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … & Polosukhin, I. (2017). Attention is all you need. _Advances in neural information processing systems_, _30_.](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

매우 많이 참고 : [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/31379)

## 시작하며

RNN 모델들은 태생적으로(?) 데이터를 Sequential 하게 차례대로 처리하기 때문에 병렬 계산이 불가능하다. 그런데 Attention 구조는 시퀀스 모델링에 있어서 거리에 구애 받지 않고 모델링 할 수 있다. **어텐션을 RNN의 보정도구로서가 아니라 어텐션만으로 인코더와 디코더를 만들면 어떨까?**\
제안하는 Transformer는 차례대로 계산하는 것을 하지 않고 Attention만 사용한다. Attention 구조를 사용함으로써 병렬계산이 가능해지고, 번역 성능도 높아진다.

$$d_{model}=512$$ : 트랜스포머의 인코더와 디코더에서 정해진 입력과 출력의 크기라고 하자.

## 하나의 인코더층

<figure><img src="../.gitbook/assets/image (30).png" alt=""><figcaption></figcaption></figure>

이것을 `num_layer`만큼 쌓는다. 논문에서는 `num_layer`=6. 인코딩하기 위해 위 그림의 인코더를 6번 지난다. 그 결과로 얻는 것이 512차원의 벡터. 그림에서 볼 수 있는 것처럼 인코더 안에는 셀프어텐션과 FFNN 두 가지의 주요 과정이 있다.

## 셀프어텐션(인코더에서 1개, 디코더에서 2개 쓰임)

어텐션 정리했던 글에서는 $$t$$시점에서의 계산만 했지만, 전체 시점에 대해서 일반화 해보자.

셀프어텐션은 입력 문장 내 단어들끼리 유사도를 구하므로 멀리 떨어진 it과 cat이 관련되어 있다는 사실 등을 학습할 수 있다.

셀프어텐션은 Q, K, V가 모두 동일하다. 정확히는 출처가 동일하다. 일반적인 seq2seq에서 사용하는 어텐션과 셀프어텐션을 비교해보면 다음과 같다.

{% hint style="info" %}
```
Seq2Seq
Q : t 시점의 디코더 셀에서의 은닉 상태
K : 모든 시점의 인코더 셀의 은닉 상태들
V : 모든 시점의 인코더 셀의 은닉 상태들

셀프어텐션
Q : 입력 문장의 모든 단어 벡터들
K : 입력 문장의 모든 단어 벡터들
V : 입력 문장의 모든 단어 벡터들
```
{% endhint %}

Q, K, V에 대해서는 [이전에 정리한 바](attention-2014.md#undefined-2) 있다.

각 단어의 $$d_{model}=512$$차원의 벡터들을 가지고 할까? 아니다. $$d_{model}=512$$차원의 단어 임베딩에서 64차원의 Q, K, V를 만들어서 사용한다. 왜 64차원일까? 셀프어텐션을 병렬적으로 `num_heads`=8개 진행하기로 하였기 때문에 나중에 가서 합치면 512차원이 되도록 하기 위함이다. 단어 임베딩 512차원 벡터에서 64차원으로 만드는 것은 행렬곱을 하면 된다.

<img src="../.gitbook/assets/image (10).png" alt="" data-size="original">  한 단어당 Q, K, V 세 가지의 벡터를 얻는다.

그리고 어텐션이 적용된다. 일반적인 어텐션은 Query와 Key를 곱하고 → 어텐션 값들 → 소프트맥스 →이 값(유사도라고 봐도 되고 확률이라고 봐도 되는 값)을 Value와 가중합 = 컨텍스트 벡터.인데, 트랜스포머 논문에서는 Query와 Key를 곱하고 → $$\sqrt{d_k}$$로 나눠준다. $$d_k$$는 Key벡터의 차원이다.

<figure><img src="../.gitbook/assets/image (3) (2).png" alt=""><figcaption><p>이 과정을 scaled dot product Attention이라고 한다.</p></figcaption></figure>

## 멀티헤드 어텐션

<figure><img src="../.gitbook/assets/image (18).png" alt=""><figcaption></figcaption></figure>

어텐션 과정을 병렬로 여러개 처리하면, 각각의 어텐션은 다른 관점으로 정보를 수집할 수 있어 효과적이다.



