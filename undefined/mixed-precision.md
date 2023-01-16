# Mixed Precision

## 일반적인 학습

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



### [Typical Mixed Precision Training](https://pytorch.org/docs/stable/notes/amp\_examples.html#id2)

<pre class="language-python"><code class="lang-python">model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

<strong><a data-footnote-ref href="#user-content-fn-1">scaler = GradScaler()</a>  # &#x3C;1>
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
3. loss를 scaling한다. scaled loss가 backward()되어 scaled gradients가 생성된다. autocast하에서의 역전파는 권장되지 않는다.
4. scaler.step은 가장 일단 optimizer에 할당된 파라미터의 gradients를 unscale한다. 만약 gradients가 infs 또는 NaNs이 아니라면 optimizer.step()이 실행되고, 그렇지 않으면 optimizer.step()은 생략된다.
5. 다음 iteration을 위해 scale을 업데이트한다.

### [Working with Unscaled Gradients](https://pytorch.org/docs/stable/notes/amp\_examples.html#id3)

#### [Gradient clipping](https://pytorch.org/docs/stable/notes/amp\_examples.html#id4)

<pre class="language-python"><code class="lang-python">scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)
            
        scaler.scale(loss).backward()
<strong>        scaler.unscale_(optimizer)
</strong><strong>        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
</strong>        scaler.step(optimizer)
        scaler.update()
</code></pre>













c\


[^1]: 학습 시작 부분에서 GradScaler를 생성한다.
