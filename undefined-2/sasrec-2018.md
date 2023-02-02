# SASRec(2018,11)(작성중)

## Paper

[Kang, W. C., & McAuley, J. (2018, November). Self-attentive sequential recommendation. In _2018 IEEE international conference on data mining (ICDM)_ (pp. 197-206). IEEE.](https://arxiv.org/pdf/1808.09781v1.pdf)

[paperswithcode](https://paperswithcode.com/paper/180809781)

## I. INTRODUCTION

순차적 추천시스템의 목적은 개인화된 모델을 최근 액션에 기반한 '문맥(context)'의 개념과 결합하는 것이다. 순차적인 동적 정보로부터 유용한 패턴을 캐치하는 것은 어려운데, 가장 큰 이유는 input space의 차원이 문맥으로 사용되는 과거 액션의 수에 exponential하게 증가하기 때문이다. 그래서 순차적 추천 분야 연구는 이러한 고차원의 동적 정보를 어떻게 간단명료하게 잡아내느냐가 관건이다.

## II. RELATED WORK

### A. General Recommendation

### B. Temporal Recommendation

### C. Sequential Recommendation

### D. Attention Mechanisms

## III. METHODOLOGY

순차적 추천에서는 유저의 액션 시퀀스 $$\mathcal{S}^u=(\mathcal{S}_1^u, \mathcal{S}_2^u, \cdots,\mathcal{S}_{|\mathcal{S}^u|}^u)$$가 주어지며 다음 아이템을 예측해야한다. 학습 시, 시간 $$t$$에서 모델은 이전의 $$t$$개의 아이템에 기반하여 다음 아이템을 예측한다.&#x20;

<figure><img src="../.gitbook/assets/image (15) (2).png" alt=""><figcaption><p>Figure 1: SASRec의 학습 과정 다이어그램. 각 time 스텝에서 모델은 모든 이전의 아이템을 고려하여 다음 액션과 관련된 아이템에 '주목'하기 위해 어텐션을 이용한다.</p></figcaption></figure>

Figure 1에서 보는 것처럼 모델의 인풋을 $$(\mathcal{S}_1^u, \mathcal{S}_2^u, \cdots,\mathcal{S}_{|\mathcal{S}^u|-1}^u)$$로, 아웃풋은 $$(\mathcal{S}_2^u, \mathcal{S}_2^u, \cdots,\mathcal{S}_{|\mathcal{S}^u|}^u)$$로 생각하면 편하다. 이번 섹션에서는 임베딩 레이어, 셀프어텐션 블록, 예측 레이어를 통해 순차적 추천시스템 모델을 어떻게 구축할 수 있는지 설명한다.

### A. Embedding Layer



### B. Self-Attention Block

### C. Stacking Self-Attention Blocks

### D. Prediction Layer

### E. Network Training

### F. Complexity Analysis

### G. Discussion

## IV. EXPERIMENTS

### A. Datasets

### B. Comparison Methods

### C. implementation Details

### D. Evaluation Metrics

추천 성능을 평가하기 위해 Hit@10과 NDCG@10을 사용했다\[14, 19]. Hit@10은 ground-truth(실제 다음 아이템)아이템이 top 10 아이템 중에 포함된 횟수를 카운트하고, NDCG@10은 높이 랭크될수록 높이 평가되는 랭크를 고려한 메트릭이다. **각 유저에 대해서 하나의 테스트 아이템밖에 없기 때문에**, Hit@10은 Recall@10과 동일하고, Precision@10을 비율로 나타낸 것과 동일하다.

아이템-유저 페어에 대한 계산이 무거워지는 것을 피하기 위하여 \[14, 48]의 전략을 따랐다. 각 유저 $$u$$에 대하여, 100개의 네거티브 아이템을 샘플링하고, 이 아이템들과 ground-truth 아이템을 함께 랭킹한다. 이렇게 101개의 아이템을 랭킹함으로써 Hit@10과 NDCG@10이 계산된다.

### E. Recommendation Performance

### F. Ablation Study

### G. Training Efficiency & Scalability

### H. Visualizing Attention Weights



\[1] S. Rendle, C. Freudenthaler, and L. Schmidt-Thieme, “Factorizing personalized markov chains for next-basket recommendation,” in WWW, 2010.

\[14] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T. Chua, “Neural collaborative filtering,” in WWW, 2017.

\[19] R. He, W. Kang, and J. McAuley, “Translation-based recommendation,” in RecSys, 2017.

\[21] R. He and J. McAuley, “Fusing similarity models with markov chains for sparse sequential recommendation,” in ICDM, 2016.

\[48] Y. Koren, “Factorization meets the neighborhood: a multifaceted collaborative filtering model,” in SIGKDD, 2008.



