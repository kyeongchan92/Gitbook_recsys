# Mixed Precision(+pytorch)

## Paper

[Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., ... & Wu, H. (2017). Mixed precision training. _arXiv preprint arXiv:1710.03740_.](https://arxiv.org/pdf/1710.03740.pdf)

## Abstract

메모리 사용량을 절반 가까이 줄이고 최신 GPU에서는 산술연산도 빠르게 한다. Weights, activations, gradients를 IEEE half precision 포맷으로 저장한다.

<figure><img src="../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

논문은 gradient까지 float16이라고 했지만, pytorch 공식 홈페이지는 backward 계산 시에는 casting을 권장하지 않는다.

논문을 읽다보면 weight, weight gradients, activation, activation gradients라는 단어가 나온다. weight와 weight gradient는 각각 모델의 파라미터와 그 파라미터의 그래디언트 값이라는건 알겠는데.. activation gradient란 뭘까? activation 함수는 Sigmoid나 ReLU처럼 그냥 통과시키는, 파라미터가 없는 함수인 줄 알았는데 activation이라는 파라미터를 가리키는 것 같은 단어도 나오고, 그래디언트까지 가졌다고 하니 처음엔 이해가 되지 않았다.

[What are "Activations", "Activation Gradients", "Weights" and "Weight Gradients" in Convolutional Neural Networks?](https://stackoverflow.com/questions/57038055/what-are-activations-activation-gradients-weights-and-weight-gradients)

위 글을 읽으면 이에 대한 답을 누가 달아놨다. Activation Gradients: This name is confusing but it is basically the **gradient of the loss** with respect to the input of the current layer $$l$$.





## 일반적인 학습

{% code lineNumbers="true" %}
```python
import copy
import time

criterion = nn.CrossEntropyLoss()
lr = 5.0  # 학습률(learning rate)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module) -> None:
    model.train()  # 학습 모드 시작
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        batch_size = data.size(0)
        if batch_size != bptt:  # 마지막 배치에만 적용
            src_mask = src_mask[:batch_size, :batch_size]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
```
{% endcode %}



### [Typical Mixed Precision Training](https://pytorch.org/docs/stable/notes/amp\_examples.html#id2)

<pre class="language-python" data-line-numbers><code class="lang-python">model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

<strong>scaler = GradScaler()  # &#x3C;1>
</strong>
for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
<strong>        with autocast(device_type='cuda', dtype=torch.float16):  # &#x3C;2>
</strong><strong>            output = model(input)
</strong><strong>            loss = loss_fn(output, target)
</strong>
<strong>        scaler.scale(loss).backward()  # &#x3C;3>
</strong><strong>        scaler.step(optimizer)  # &#x3C;4>
</strong><strong>        scaler.update()  # &#x3C;5>
</strong></code></pre>

1. 학습 시작 부분에서 GradScaler를 생성한다.
2. autocasting과 함께 순전파 한다.
3. loss를 scaling한다. scaled loss가 backward()되어 scaled gradients가 생성된다. autocast하에서의 역전파는 권장되지 않는다. **참고로 loss\_fn으로 구한 loss는 float32이다. pytorch의 loss객체는 계산할 때 float32로 바꾼다고 한다.**
4. scaler.step은 가장 일단 optimizer에 할당된 파라미터의 gradients를 unscale한다. 만약 gradients가 infs 또는 NaNs이 아니라면 optimizer.step()이 실행되고, 그렇지 않으면 optimizer.step()은 생략된다.
5. 다음 iteration을 위해 scale을 업데이트한다.

### [Working with Unscaled Gradients](https://pytorch.org/docs/stable/notes/amp\_examples.html#id3)

#### [Gradient clipping](https://pytorch.org/docs/stable/notes/amp\_examples.html#id4)

<pre class="language-python" data-overflow="wrap" data-line-numbers><code class="lang-python">scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)
            
        scaler.scale(loss).backward()
<strong>        scaler.unscale_(optimizer)  # &#x3C;1>
</strong><strong>        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # &#x3C;2>
</strong>        scaler.step(optimizer)  # &#x3C;3>
        scaler.update()  # &#x3C;4>
</code></pre>

1. optimizer에 할당된 파라미터의 gradients를 unscale한다(in-place 방식).
2. 1번을 수행했기 때문에 평소처럼 clipping을 수행한다.
3. 만약 gradient가 infs나 NaNs를 포함하고 있어서 optimizer.step()을 생략하더라도, 1번을 수행했기 때문에 scaler.step()은 optimizer에 할당된 파라미터의 gradient를 unscale하지 않는다.
4. 다음 iteration을 위해 scale을 업데이트한다.





## Ref

{% embed url="https://hoya012.github.io/blog/Mixed-Precision-Training/" %}

[Docs ](https://pytorch.org/docs/stable/index.html)> [CUDA Automatic Mixed Precision examples](https://pytorch.org/docs/stable/notes/amp\_examples.html)

[Tutorials ](https://pytorch.org/tutorials/index.html)> [PyTorch Recipes](https://pytorch.org/tutorials/recipes/recipes\_index.html) > [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp\_recipe.html)

{% embed url="https://effectivemachinelearning.com/PyTorch/8._Faster_training_with_mixed_precision" %}
