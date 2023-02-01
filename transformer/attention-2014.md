# Attention(2014) 복습

## 원논문

[Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. _arXiv preprint arXiv:1409.0473_.](https://arxiv.org/abs/1409.0473)

밑바닥부터 시작하는 딥러닝2를 정리하였습니다. 그림은 직접 그렸습니다.

## 한 줄 요약

$$h$$대신 $$hs$$를 디코더에 전달한다.

사람이 번역할 때처럼 대응하는 단어에 주목하여 번역한다.

## Seq2seq의 문제 <a href="#5bee" id="5bee"></a>

<figure><img src="../.gitbook/assets/image (2) (1) (4).png" alt=""><figcaption><p>가장 기본적인 Sequence to Sequence.</p></figcaption></figure>

Seq2seq는 입력 문장이 길든 짧든 **고정 길이**의 벡터에 인풋 시퀀스의 모든 정보를 압축한다. 인코더의 출력이 항상 같은 길이의 벡터일 필요가 있을까? 이를 개선하는 것이 Attention의 첫 번째 포인트다. 인코더의 아웃풋 벡터를 입력 시퀀스의 길이에 따라 바꿔주자. 이를 해결할 수 있는 방법은 바로 인코더의 모든 hidden state를 이용하는 것이다.

<figure><img src="../.gitbook/assets/image (29).png" alt=""><figcaption></figcaption></figure>

기존 인코더에서는 마지막 hidden state만을 디코더에 넘겨줬다면, 단지 각 스텝의 모든 hidden state를 넘겨준다는 것 만으로도 고정 벡터 문제는 얼추 해결된 것으로 보인다. 이제 디코더가 $$hs$$를 어떻게 활용할 것인지 보면 될 것이다.

우선 기존 디코더를 생각해보자.

<figure><img src="../.gitbook/assets/image (7) (4).png" alt=""><figcaption><p>기존 디코더</p></figcaption></figure>

기존 디코더는 인코더에서 넘어온 hidden representation, $$h$$를 첫 번째 hidden state로 사용했다. 참고로 인코더의 인풋 시퀀스는 ‘나는 고양이로소이다’처럼 번역 전의 시퀀스이고, 디코더의 인풋 시퀀스는 ‘i am a cat’처럼 번역 후의 시퀀스이다. Softmax의 출력은 모든 토큰에 대한 확률이다.

이제 개선을 위해 $$h$$대신 $$hs$$를 디코더에 전달하자.

그리고 사람이 번역할 때처럼 ‘나 — i’ 또는 ‘고양이 — cat’ 같이 어디에 주목해야 할 지도 학습 시켜 보자. 즉, 입력과 출력에서 서로 어디에 관련되어 있는 지를 학습 시켜 보자. 수동이 아니라 자동으로 말이다. 이렇게 $$hs$$를 전달하는 것, 어디에 주목 해야 하는지 계산하는 과정이 바로 Attention구조이다.

<figure><img src="../.gitbook/assets/image (22).png" alt=""><figcaption><p>Attention이 추가된 디코더(아직은 간단하게만)</p></figcaption></figure>

인코더의 마지막 hidden state가 디코더의 첫 번째 hidden state로 쓰이는 것은 기존 디코더와 동일하다. 그러나 디코더의 각 hidden state와 $$hs$$를 가지고 Attention 계산하는 것이 추가되었다. 저 빨간 구름 안의 연산을 살펴보자.

빨간 구름 안에서는 크게 두 가지 과정이 존재한다. 바로 Attention Weight와 Weight Sum이 그것이다.

## 1. Attention Weight 과정 (a 구하기) <a href="#e6d2" id="e6d2"></a>

<figure><img src="../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>

_‘\<eos>’_라는 단어가 들어갈 때는 $$hs$$중에서 ‘나’라는 hidden state를 선택하여, 같이 사용해 예측하고자 한다. 그런데 선택이라는 작업은 미분이 불가능(=역전파 불가능, 학습 불가능)하기 때문에, 가중합을 하여 ‘선택’이라는 작업을 대체한다.

이제부턴 디코더로부터 어느 시점에 나온 hidden representation을 $$h$$라고 하자. 기존에는 $$h$$만 이용해서 다음 단어를 예측했다면, 이제 인코더로부터 넘어온 $$hs$$가 있으니, $$hs$$ 중에서 어떤 것에 주목해야 하는지 골라보자.

$$hs$$와 $$h$$를 내적하면 $$s$$라는 스코어가 나온다. 이는 $$hs$$와 $$h$$ 사이의 유사도이다. 이를 소프트맥스 층에 통과시키면 0\~1사이의 값으로 표현된다. 위의 예에서는 $$h$$와 ‘나’라는 벡터가 가장 유사도가 높게 나왔다. 이 수치들을 **Attention Weight**, $$a$$ 벡터라고 하자.

## 2. Weight Sum 과정(context vector, **c**구하기) <a href="#30c2" id="30c2"></a>

$$a$$를 구했다. $$a$$는 ‘$$h$$가 $$hs$$ 중에서 어디에 주목해야 하는지’ 를 알려준다. $$a$$가 의미하는 것을 다시 찬찬히 짚어볼까? 지금 디코더의 hidden state는 \<eos>에 대한 $$h$$이다. 이 $$h$$는 $$hs$$ 중 어디에 주목해야 할까?

디코더의 $$h$$는 인코더의 ‘나’라는 벡터에 주목해야하지 않을까? 그래야 결과가 ‘_i_ ’가 나올테니 말이다.

지금은 인위적으로 $$h$$와 인코더의 ‘나’ 벡터 간의 유사도가 0.8이라고 해보자. 그리고 다른 $$hs$$들과는 유사도가 좀 낮다고 해보자. 이 유사도들이 모여있는게 $$a$$다. 다시말해, $$a$$는 ‘$$h$$가 $$hs$$ 중에서 어디에 주목해야 하는 지’ 를 알려준다.

그럼 이 $$a$$와 인코더의 $$hs$$를 곱하면? 인코더의 ‘나’ 벡터를 선택하는 것과 비슷한 작업이 된다. ‘나’ 벡터가 80%나 함유돼있기 때문이다. ‘선택’이라는 작업을 가중합으로 수행하는 것이다. 그리고 이 벡터들을 다 더하면 context vector를 구할 수 있다. 그림으로 보면 다음과 같다.

<figure><img src="../.gitbook/assets/image (1) (1) (7).png" alt=""><figcaption><p>가중합을 계산하여 Context vector를 구한다.</p></figcaption></figure>

‘나’에 해당하는 가중치가 0.8이었기 때문에, $$c$$에는 ‘나’ 벡터의 성분이 많이 포함되어 있을 것이다(색깔도 비슷하게 맞췄다). 이로써 예측을 수행할 때, ‘나’라는 특정 단어에 주목하는 context vector를 얻었다. 이제 $$h$$ 혼자로만 예측하는 것이 아니라 context vector까지 협력하여 예측할 것이다.

<figure><img src="../.gitbook/assets/image (28) (1).png" alt=""><figcaption></figcaption></figure>

이렇게 얻은 context vector와 $$h$$를 함께 concat하여 Affine 계층에 넘겨준다. Affine 계층에 RNN의 hidden state 뿐만 아니라 Attention 계층의 context vector까지 더해지게 되었다. 디코더 안에서의 계산 과정을 정리하면 위 그림과 같다.

1. Attention Weight : **h**와 **hs**로 **a**를 계산한다.
2. Weight Sum : **a**와 **hs**로 **c**를계산한다.
3. **c**와 **h**를 concat해서 Affine 계층에 입력해준다.

끝!

## [**딥러닝을 이용한 자연어 처리 입문**](https://wikidocs.net/22893) **페이지에서 설명하는 방식**

Query, Key, Value 구조 먼저 정의.

**쿼리(query)**가 주어지면, 모든 **키(key)**와의 유사도를 구하여 이 유사도를 키와 매핑된 **값(value)**에 반영한 후, 값(value)을 모두 더해 리턴함. 이 결과 값이 어텐션 값(Attention value)이다.

<figure><img src="../.gitbook/assets/image (12) (1).png" alt=""><figcaption></figcaption></figure>

어텐션을 위 구조와 대응시켜보면,

**t-시점의 디코더 셀에서의 h(query)**가 주어지면, 모든 **인코더의 hs들(key)**와의 유사도를 구하여 이 유사도를 키와 매핑된 **인코더의 hs들(value)**에 반영한 후, 값(value)을 모두 더해 리턴함.

즉, 어텐션에서는 키와 밸류가 동일하다.

