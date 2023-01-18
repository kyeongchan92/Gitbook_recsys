# Loss의 reduction 옵션

##

## &#x20;출처

{% embed url="https://stats.stackexchange.com/questions/358786/mean-or-sum-of-gradients-for-weight-updates-in-sgd" %}

## 궁금증

BCELoss를 정리할 때, Pytorch의 Loss객체에는 reduction이라는 옵션이 있었던 걸 알 수 있다.

{% embed url="https://app.gitbook.com/o/9pPdlAX9tD5yWurWffl2/s/DrsfkmCAUYPdnNOPwzWG/~/changes/AuQK0TVBMDhYBYnQhCN5/undefined/bceloss" %}

디폴트 값은 mean이지만 sum으로 사용할 수도 있는데 이 둘의 차이는 뭘까? 혹시 loss를 더한 것과 평균낸 것이 학습에 다른 영향을 미치는 것은 아닐까? 결론부터 말하자면 이 논의는 '별로 중요하지 않다.'

## 왜 안중요?

loss를 미분하여 Gradient $$G$$를 얻는다. 평균을 내든 더하든 **Loss는 스칼라값이다.** Gradient는 다음과 같이 얻어진다. $$f$$는 loss fuction이고 $$x_i$$는 인풋 데이터이다.

$$
G = \nabla\sum_{i=1}^n  f(x_i)
$$

$$n$$은 배치 사이즈이다. 우린 많은 경우 배치로 돌릴 것이기 때문이다. 여기까지 하면 gradient란 loss를 모든 배치 샘플에 대해 더한 후 미분한 것이다. 일단 나누지 말고 여기서 파라미터 업데이트를 해보자.

파라미터 업데이트 시간이 왔다. SGD같은 경우에는 아래 식과 같이 업데이트된다.

$$
x^{(t+1)} = x^{(t)}- r G
$$

파라미터 업데이트 시 learning rate($$r$$)과 $$G$$를 이용한다. 별 다를 것 없이 우리가 이미 알고 있는 수식이다. 근데 만약 mean 처리 했다면?

$$
x^{(t+1)} = x^{(t)}- \frac{\tilde{r}}{n} G.
$$

위와 같은 식이 될 것이다. 단지 learning rate를 조절해주기만 하면 된다! learning rate는 사용자에 의해 최적 값으로 튜닝될 것이므로, 평균을 취하나 더하나 별로 중요하지 않다. 만약 배치사이즈가 10배 크게 해서 $$G$$가 커졌다면, learning rate를 10으로 나눠주면 똑같다. 다만, 평균을 선호하는 이유는 learning rate와 배치사이즈를 분리할 수 있기 때문이다.





