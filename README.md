# Netflix 추천 엔진

이 프로젝트는 Netflix에서 유저의 이전 시청 선호도와 평점을 기반으로 콘텐츠를 제안하도록 설계된 머신러닝 기반 추천 시스템입니다. 협업 필터링 기술과 클러스터링의 조합을 사용하여 시스템은 유저 선호도를 예측하여 맞춤형 추천을 제공합니다.

전체 결과 및 분석은 여기에서 확인할 수 있습니다: [Netflix 분석](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Results%20copy.ipynb)

사용 기술 스택: Python, NumPy, Pandas, scikit-learn, Seaborn, Matplotlib, Yellowbrick, Cleanlab

For English version, please click here: [Overview in English](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/README%20(EN).md)

## 목차
- [프로젝트 개요](#프로젝트-개요)
- [데이터셋 및 방법론](#데이터셋-및-방법론)
- [접근 방식](#접근-방식)
- [평가](#평가)
- [결과](#결과)
- [향후 개선 사항](#향후-개선-사항)

## 프로젝트 개요
이 Netflix 추천 시스템은 협업 필터링 기반으로 유저가 아직 시청하지 않은 콘텐츠에 대한 평점을 예측하여 맞춤형 추천을 생성합니다.
유사한 유저를 클러스터링하고 확률적 모델링을 적용함으로써 유저 선호도의 잠재적 구조를 발굴하고, 개인화된 추천을 통해 유저 참여도를 향상시키는 것을 목표로 합니다.

- 문제 개요

    - 추천 시스템은 Netflix와 같은 스트리밍 플랫폼에서 고객 유지(Customer Retention)에 핵심적인 역할을 합니다.
    유저 취향에 맞는 콘텐츠를 지속적으로 제공함으로써 시청 경험을 개선하고, 이는 곧 구독 유지 및 매출과 직결됩니다.
    반대로 추천 품질이 낮을 경우 유저 경험이 저하되어 이탈률 증가로 이어질 수 있습니다.

    - 프로덕션 수준의 추천 시스템을 구축하기 위해서는 먼저 유저 선호도의 기본적인 분포 구조를 이해하는 것이 필수적입니다.
    유저 행동을 모델링함으로써 각 콘텐츠에 대한 선호 확률을 추정할 수 있으며, 이를 통해 유저가 해당 콘텐츠를 즐길 가능성을 예측할 수 있습니다.

    - 본 프로젝트에서는 데이터셋을 기반으로 유저 평점 데이터의 구조를 분석한 뒤,
    추천을 위한 유저 평점 예측 알고리즘을 정의하고 비교 및 검증함으로써 추천 성능이 가장 우수한 모델을 도출하는 것을 목표로 합니다.

## 데이터셋 및 방법론

데이터셋은 유저 평점 정보를 포함하고 있으며, 1,200 × 1,200 크기의 유저–아이템 평점 행렬로 구성되어 있습니다.
누락된 평점 값은 0으로 표시되어 있으며, 이러한 결측값을 보완하기 위해 K-means 및 가우스 혼합 모델(GMM)을 활용한 클러스터링과 기대값 최대화(Expectation–Maximization, EM) 알고리즘을 통한 평점 예측을 수행했습니다.

K-means는 계산 효율성, 단순성, 해석 가능성 측면에서 장점이 있어 초기 baseline 모델로 선택되었습니다.
대규모 데이터셋에서도 안정적으로 동작하며 명확한 hard clustering 할당을 통해 평점 패턴에 기반한 주요 유저 세그먼트를 빠르게 식별할 수 있습니다. 이를 통해 유저 segmentation의 기준선을 설정하고 RMSE 및 실루엣 점수와 같은 지표를 활용하여 군집화 구조를 평가할 수 있었습니다.

그러나 Netflix와 같은 실제 스트리밍 플랫폼 환경에서 유저 선호도는 단일한 그룹에 독점적으로 속하지 않는 경우가 많습니다.
대부분의 유저는 여러 취향 그룹에 동시에 속하며 이러한 특성은 hard clustering만으로는 충분히 반영하기 어렵습니다.
이를 보완하기 위해 GMM을 도입하였습니다.

GMM은 각 클러스터를 확률 분포로 모델링하고 유저에게 소프트 할당 확률을 배정함으로써 선호도의 중첩과 불확실성을 효과적으로 표현할 수 있습니다. 이러한 확률적 접근 방식은 실제 시청 행동과 부합하며 보다 정교한 추천을 가능하게 합니다. GMM을 통해 각 데이터 포인트에 대한 클러스터 확률이 추정된 이후 (EM) 알고리즘을 적용하여 평점을 예측합니다.
본 추천 시스템에서 EM 알고리즘은 유저 간 선호도의 유사성을 확률적으로 학습함으로써 평점 예측의 정확도를 향상시키는 역할을 합니다.

1. **E-단계 (기대값 단계)**: 관찰된 데이터와 현재 매개변수 추정치를 기반으로 잠재 변수의 기대값을 계산합니다.

   데이터 $X$와 현재 매개변수 $\theta$가 주어지면 각 클러스터 $k$에 대한 사후 확률 (posterior probabilities)을 계산합니다:
   
   $\gamma_{z_i} = P(z_i | X; \theta) = \frac{\pi_k \, \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_i | \mu_j, \Sigma_j)}$
   
   여기서:
   - $\gamma_{z_i}$는 데이터 포인트 $x_i$에 대한 클러스터 \( k \)의 사후 확률이며,
   - $\pi_k$는 클러스터 $k$의 사전 확률이고,
   - $\mathcal{N}(x_i | \mu_k, \Sigma_k)$는 평균 $\mu_k$와 공분산 $\Sigma_k$를 가진 클러스터 $k$의 가우스 분포입니다.

3. **M-단계 (최대화 단계)**: E-단계의 기대 로그 우도 (log likelihood)를 최대화하여 매개변수 $\theta = (\pi_k, \mu_k, \Sigma_k)$를 업데이트합니다.

   매개변수는 다음과 같이 업데이트됩니다:
   - **평균 업데이트**:
     
     $\mu_k = \frac{\sum_{i=1}^{N} \gamma_{z_i} \, x_i}{\sum_{i=1}^{N} \gamma_{z_i}}$
     
   - **공분산 업데이트**:
     
     $\Sigma_k = \frac{\sum_{i=1}^{N} \gamma_{z_i} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^{N} \gamma_{z_i}}$
     
   - **혼합 가중치 업데이트**:
     
     $\pi_k = \frac{1}{N} \sum_{i=1}^{N} \gamma_{z_i}$
     
   여기서 $N$은 총 데이터 포인트 수이고 $K$는 클러스터 수입니다.

이러한 E, M 단계는 수렴할 (convergence) 때까지 반복되어 유저 평점 예측의 정확도를 향상시키는 최적화된 매개변수 $\theta$를 생성합니다.

- 성능 비교: K-Means vs GMM

| K | K-Means RMSE | K-Means BIC | K-Means LL | GMM RMSE | GMM BIC | GMM LL |
|---|---|---|---|---|---|---|
| 1 | 0.5667 | -2,821,274 | -2,817,016 | 0.4802 | -1,544,758 | -1,540,500 |
| 2 | 0.4805 | -2,707,332 | -2,698,814 | 0.4559 | -1,464,771 | -1,456,252 |
| 3 | 0.4846 | -2,676,802 | -2,664,022 | 0.4566 | -1,468,236 | -1,455,457 |
| 4 | 0.4746 | -2,653,928 | -2,636,887 | 0.4756 | -1,447,234 | -1,430,193 |
| 5 | 0.4873 | -2,642,527 | -2,621,225 | 0.4738 | -1,450,889 | -1,429,587 |
| 6 | 0.4961 | -2,636,149 | -2,610,585 | 0.4725 | -1,453,663 | -1,428,099 |
| 7 | 0.4940 | -2,631,491 | -2,601,667 | 0.4759 | -1,453,335 | -1,423,511 |
| 8 | 0.5025 | -2,626,356 | -2,592,271 | 0.4920 | -1,445,582 | -1,411,497 |
| 9 | 0.4959 | -2,622,983 | -2,584,636 | 0.4967 | -1,449,051 | -1,410,704 |
| 10 | 0.4800 | -2,619,471 | -2,576,864 | 0.4974 | -1,452,497 | -1,409,889 |
| 11 | 0.4873 | -2,618,841 | -2,571,972 | 0.4998 | -1,450,890 | -1,404,022 |
| 12 | 0.4797 | -2,618,970 | -2,567,840 | 0.5020 | -1,454,027 | -1,402,897 |

- 유저 평점 데이터를 기반으로 한 실험 결과, GMM이 K-means 대비 소폭 더 나은 성능을 보였으나 현재 데이터셋에서는 두 모델 모두 전반적으로 유사한 성능을 나타냈습니다.
성능 차이는 제한적이었으며 현 결과만을 기준으로 할 때 두 모델 모두 합리적인 선택으로 판단됩니다. 다만, GMM은 확률적 클러스터링을 통해 유저 선호도의 중첩과 불확실성을 모델링할 수 있으므로 보다 복잡한 데이터 분포에서는 K-means보다 우수한 일반화 성능을 보일 가능성이 있습니다. 이러한 가설은 추가적인 실험을 통해 검증이 필요합니다.

### 가설
- 현재 데이터셋에서는 K-means와 GMM이 유사한 성능을 보일 것으로 예상되나, 데이터 분포의 복잡성이 증가할수록 GMM이 K-means보다 더 나은 일반화 성능을 보일 것으로 가정합니다. 해당 가설을 검증하기 위해 EDA 및 하이퍼파라미터 최적화 이후 K-means와 GMM을 3가지 스트레스 테스트 환경에서 비교 평가합니다.
- 가설을 테스트하기 위해 EDA를 수행하고 매개변수를 최적화한 후 GMM 및 K-means 모델을 다음과 같은 3가지 스트레스 테스트를 통해 평가합니다:
  - 스트레스 테스트 1: 다양한 비율의 누락된 데이터 [0.1, 0.3, 0.5, 0.7]
  - 스트레스 테스트 2: 노이즈 도입 [baseline, 보통, 강함, 적대적]
  - 스트레스 테스트 3: 80% 누락된 데이터 + 희소 / 롱테일 이상치 (sparse / heavy-tailed outlier noise) 도입
- 스트레스 테스트 3의 결과는 최적 GMM의 백분율 개선을 계산하는 데 사용됩니다.

## 접근 방식

### 탐색적 데이터 분석(EDA)

기본 데이터 구조를 분석하기 위해 K-means와 GMM 모두에 대해 t-SNE 시각화를 수행합니다. 이를 위해 먼저 t-SNE의 주요 하이퍼파라미터인 (hyperparameter) perplexity 값을 결정해야 합니다.
Perplexity 설정은 KL 발산(Kullback–Leibler Divergence)을 기준으로 최적화합니다.
또한 K-means의 경우, 군집화 품질 평가를 위해 왜곡(inertia, distortion) 및 실루엣 점수(silhouette score)를 추가로 계산합니다.

| K-means 왜곡 | K-means 실루엣 |
| -- | -- |
| ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Kmeans%20Distortion.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Kmeans%20Silhouette.png) |

| K-means | GMM |
|--|--|
| ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Kmeans%20Perplexity.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/GMM%20Perplexity.png) |
| ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Kmeans%20KL.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/GMM%20KL.png) |

- 왜곡 및 실루엣 점수는 데이터에 다양한 유저 평점 분포가 있음에도 불구하고 K-means가 낮은 K 값에서 가장 잘 수행됨을 나타냅니다.
- KL-발산 점수와 Perplexity 시각화를 기반으로 t-SNE 설정에 복잡도 = 50을 선택합니다. 

이제 K-means와 GMM에 대한 t-SNE 플롯을 시각화합니다:

<table>
  <tr>
    <td><strong>K-means</strong></td>
    <td><img src="https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Kmeans%20tSNE.png" /></td>
  </tr>
  <tr>
    <td><strong>GMM</strong></td>
    <td><img src="https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/GMM%20tSNE.png" /></td>
  </tr>
</table>

가설에서 예상했듯이 K-means 모델은 모든 K 값에서 상당한 중첩을 보이며 전반적으로 하나의 blob로 뭉치는 경향을 보입니다. 이는 Netflix 유저 평점 데이터의 잠재적 구조를 K-means가 효과적으로 일반화하지 못함을 시사합니다. 반면, GMM 모델은 각 그룹 간의 명확하고 뚜렷한 분리를 보입니다. 이러한 시각화 결과는 GMM이 고차원 데이터에 대해 효과적으로 일반화한다는 점을 뒷받침하며 이는 사전에 설정한 가설과 일치합니다. 이러한 관찰을 보다 확실히 검증하기 위해 두 모델 모두 스트레스 테스트를 수행할 필요가 있습니다.

먼저 비교를 위해 결과를 시각화합니다:

| RMSE | GMM BIC | K-means BIC |
| -- | -- | -- |
| ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/RMSE%20plots.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/GMM%20BIC.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Kmeans%20BIC.png) |

### 모델 선택

이러한 결과를 바탕으로, K-means와 GMM의 RMSE 값은 전반적으로 안정적이며 유사한 수준을 유지함을 확인할 수 있습니다. 두 모델의 성능은 클러스터 수에 따라 교대로 나타나며, 특정 K 범위에서는 K-means가 GMM보다 우수한 성능을 보이고, 다른 범위에서는 GMM이 더 나은 성능을 보입니다.

두 모델 모두 본 데이터셋에서 우수한 예측 성능을 보이지만 manifold analysis 결과 K-means는 기본 데이터 분포를 효과적으로 일반화하지 못함이 관찰됩니다. 특히 K-means의 BIC 값은 K가 증가함에 따라 단조 증가하는 경향을 보이며 이는 클러스터 수가 늘어날수록 모델 성능이 오히려 악화됨을 의미합니다.

K-means 모델은 K = 3에서 최적의 성능을 달성함을 시사하며 이는 앞서 관찰된 최소 왜곡 점수(distortion score) 결과와도 일관됩니다. 해당 지점을 초과하여 클러스터 수를 증가시킬 경우 추가적인 의미 있는 구조를 포착하지 못하고 모델 성능만 저하되는 현상이 발생합니다.

이는 K-means 모델이 Netflix 유저 평점 분포 내의 미묘한 구조적 패턴을 표현하는 데 한계를 가진다는 점을 보여줍니다. 클러스터 수가 강제로 증가함에 따라 K-means는 local 구조에 과적합되는 (overfitting) 반면 전역 manifold를 일반화하지 못해 성능이 저하됩니다.

반면, GMM의 BIC 값은 모든 K 값에서 K-means보다 절대적으로 높은 값을 보입니다. 그러나 K = 3 이상에서는 GMM의 BIC가 거의 단조 감소하는 추세를 보이며 전역 최소값은 K = 20에서 관찰됩니다. 이는 BIC 페널티에도 불구하고 우도 증가에 따른 모델 개선 효과가 존재함을 의미합니다.

종합적으로 볼 때 현재 데이터 조건에서는 K-means와 GMM 모두 RMSE 기준으로 우수한 성능을 보입니다. 그러나 모델의 일반화 능력과 구조적 표현력을 보다 명확히 비교하기 위해 이후 단계에서는 스트레스 테스트를 통해 두 모델을 추가적으로 평가합니다.

위 결과를 바탕으로 스트레스 테스트 평가를 위해 다음과 같이 모델을 선정합니다:

- **K-means K = 3 (시드 = 0):** RMSE 및 BIC, 왜곡 점수를 기반으로 한 최고 성능의 K-means 설정.
- **GMM K = 3 (시드 = 1):** BIC의 두 번째로 낮은 최소값. RMSE의 전역 최소값은 K = 4에서 달성되었으나 해당 지점에서 BIC 값 또한 두 번째로 높은 수준에 도달하였습니다.
이에 따라, 모델 복잡도에 대한 페널티를 고려하여 K = 4 대신 K = 3을 테스트 대상으로 선택하였습니다.
- **GMM K = 20 (시드 = 1):** 모델 복잡도에 따른 BIC 페널티에도 불구하고 큰 우도(likelihood) 이득을 동반한 BIC의 전역 최소값.

## 평가
### 스트레스 테스트 1 (누락된 데이터)

이 스트레스 테스트에서는 다양한 비율로 데이터 누락을 강제합니다. 즉, 다음 심각도로 모델들을 테스트합니다:
- [0.1, 0.3, 0.5, 0.7]
- 위의 누락된 데이터 비율은 전체 행렬이 아닌 관찰된 데이터의 백분율을 나타냅니다.
- 따라서 누락된 데이터를 강제한 후 실제 누락된 데이터의 총 백분율은 해당 비율보다 높을 수 있습니다.

정의된 모델 매개변수를 사용하여 다음 결과를 얻습니다:
| 스트레스 테스트 1 결과 | 스트레스 테스트 1 t-SNE |
| -- | -- |
| ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Stress%20Test%201.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Stress%20Test%201%20tSNE.png) |

**요약**
- RMSE 기준으로 두 GMM 모델 모두 K-means에 비해 누락된 데이터에 훨씬 더 견고한 성능을 보입니다. 누락 비율이 증가함에 따라 K-means의 RMSE는 거의 지수적으로 (exponential) 증가하는 반면, GMM의 RMSE는 매우 완만하게 증가합니다.
- 모든 누락 데이터 비율에서 K-means의 BIC 값은 안정적으로 유지됩니다. 그러나 이는 데이터 부족으로 인한 과소적합(underfitting)의 결과입니다. 실제로 K-means의 RMSE 성능은 크게 악화되며 K가 증가할수록 과소적합이 심화되어 예측 성능이 지속적으로 저하됩니다. 즉, 충분한 데이터가 없는 환경에서는 더 나은 BIC 점수에도 불구하고 K-means는 과소적합을 극복하지 못합니다.
- GMM의 entropy는 단조 증가하지만 전반적으로 0에 가까운 값을 유지하며 이는 상당한 수준의 누락 데이터가 존재하더라도 높은 확신도의 군집화 할당이 이루어지고 있음을 의미합니다.
- K = 20에서 K-means와 GMM 간의 클러스터 일치는 사실상 관찰되지 않습니다. 이는 K-means가 의미 있는 클러스터 구조를 식별하기 어려운 조건에서 두 모델 간 군집화 값이 크게 발산됨을 (divergence) 나타냅니다.

**t-SNE 분석**
- K-means
  - 누락 비율이 증가함에 따라 클러스터가 점점 확산되고 서로 겹치며, 결국 대부분의 클러스터가 단일 blob로 수렴합니다.
  - 이는 상당한 수준의 누락된 데이터가 존재할 경우 K-means가 뚜렷한 클러스터 경계를 유지하는 데 한계를 보여줍니다.
- GMM
  - 높은 누락 비율에서도 클러스터 구조가 효과적으로 정의되고 분리된 상태로 유지됩니다.
  - 누락 비율이 증가함에 따라 클러스터 간 거리가 확대되며 클러스터 간 불확실성이 증가했음을 의미합니다. 이는 예상된 결과로 누락된 데이터가 많아질수록 GMM은 각 데이터 포인트에 대해 더 낮은 클러스터 할당 확률을 부여합니다.

또한 K = 20에서의 GMM은 K = 3에 비해 더 미묘한 클러스터 구조를 보여주며 유저 간의 세밀한 선호도 차이를 보다 효과적으로 포착합니다.

본 스트레스 테스트를 통해 GMM이 누락된 데이터가 존재하는 조건에서도 고차원 데이터 분포를 효과적으로 모델링할 수 있는 K-means 모델 보다 견고한 접근법임을 확인하였습니다.
이제 스트레스 테스트 2를 시작합니다.

### 스트레스 테스트 2 (noise 데이터 도입)

이 스트레스 테스트에서는 데이터 전반에 다양한 수준의 noise와 데이터 손상을 도입합니다. 다음 심각도로 모델들을 테스트합니다:
- baseline (이분산 가우시안, heteroskedastic Gaussian), 보통 (항목 상관 노이즈, item-correlated noise), 강함 (비무작위 누락, MNAR), 적대적 (적대적 뒤집기, adversarial flips)

정의된 모델 매개변수를 사용하여 다음 결과를 얻습니다:
| 스트레스 테스트 2 결과 | 스트레스 테스트 2 t-SNE |
| -- | -- |
| ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Stress%20Test%202.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Stress%20Test%202%20tSNE.png) |

**요약**
- K = 3인 GMM은 전반적으로 가장 낮은 RMSE를 유지하는 반면, 초기 noise 강도 구간에서는 K-means가 GMM (K = 20) 모델 보다 더 나은 성능을 보입니다.
- 강한 노이즈(Strong noise) 수준에서는 GMM이 RMSE 기준으로 특히 우수한 성능을 보입니다.
- GMM (K = 3) 모델과 K-means 간 군집화 일치도는 모든 noise 유형에서 약 30~50% 수준을 유지하는 반면, GMM (K = 20) 모델과 K-means 간 클러스터 일치도는 0~10% 수준을 유지합니다.
- GMM (K = 3) 모델의 entropy 값이 가장 낮게 유지되며 noise 강도가 증가함에 따라 entropy도 증가하는 경향을 보입니다.

**t-SNE 분석**
- 위 시각화 결과는 서로 다른 noise 유형이 클러스터 구조에 미치는 영향을 보여줍니다. 본 스트레스 테스트는 매우 극단적인 조건을 가정하며 GMM 모델과 K-means 모델 모두 강한 noise 환경에서는 결국 성능이 붕괴되는 한계를 보입니다. 다만, K-means는 기준선 수준의 noise에서도 이미 클러스터 구조가 붕괴되기 시작하며 noise 강도가 증가할수록 t-SNE 플롯에서 클러스터 분리가 악화됩니다. 반면 GMM 모델은 특히 적대적 노이즈(adversarial noise) 조건에서 모델 붕괴 이전 시점(BIC 최소 구간)까지 일정 수준의 클러스터 구조를 유지하며 K-means 대비 더 높은 견고성을 보입니다.
- 이러한 결과는 Netflix 유저 데이터의 구조적 특성을 고려할 때 GMM clustering이 K-means보다 전반적으로 더 견고함을 입증합니다. 그러나 관찰된 바와 같이 GMM 역시 누락 데이터에 완전히 견고하지 않으며, 특히 noise 및 데이터 손상 환경에서는 성능 저하가 관찰됩니다. 현재 GMM 모델은 강한 noise 스트레스 테스트에서 상당한 한계를 보였으며 이러한 문제를 극복하기 위해서는 추가적인 모델 개선과 정교한 접근이 필요함을 보여줍니다.

이제 스트레스 테스트 3을 수행합니다.

### 스트레스 테스트 3 (누락된 데이터 + 희소 / 롱테일 이상치)
이 스트레스 테스트는 실제 데이터 시나리오를 시뮬레이션하도록 설계되었습니다. 유저가 Netflix 콘텐츠를 평가할 가능성이 매우 낮으므로 높은 누락된 데이터 비율은 이를 정확하게 표현합니다. 또한, 희소하고 롱테일 이상치는 가장 인기 있는 콘텐츠만 가장 많은 평점을 받기 때문에 가장 현실적인 노이즈 데이터를 나타냅니다. 상당수의 Netflix 콘텐츠는 매우 적은 평점을 받습니다.

본 스트레스 테스트는 현실 세계 데이터 환경을 시뮬레이션하기 위해 설계되었습니다. 실제로 대부분의 유저는 Netflix 콘텐츠를 평가할 가능성이 매우 낮으므로, 높은 누락 데이터 비율은 현실적인 유저 평점 패턴을 정확히 반영합니다. 또한, 희소하고(sparse) 롱테일(long-tailed) 이상치(outlier)는 가장 현실적인 noise 유형으로, 이는 소수의 인기 콘텐츠에만 평점이 집중되는 반면 상당수의 콘텐츠는 매우 적은 평점만을 보유하기 때문입니다. 이러한 데이터 특성은 실제 Netflix 콘텐츠 소비 및 평점 분포를 정확하게 반영합니다.

정의된 모델 매개변수를 사용하여 다음 결과를 얻습니다:
| 스트레스 테스트 3 결과 | 스트레스 테스트 3 t-SNE |
| -- | -- |
| ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Stress%20Test%203.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Stress%20Test%203%20tSNE.png) |

## 결과
위의 결과는 높은 누락 데이터 비율과 noise가 존재하는 가혹한 조건에서도 GMM이 더 높은 견고성(robustness)을 보여줍니다. 이러한 결과는 이전 테스트들과 일관되며, GMM (K = 3) 모델은 RMSE와 BIC 측면에서 K-means보다 우수한 성능을 보인 반면, GMM (K = 20) 모델은 K-means 대비 더 열등한 RMSE와 BIC를 기록했습니다. 또한, GMM (K = 20) 모델의 entropy는 K = 3 보다 높게 나타났으며, 이는 클러스터 할당에서의 불확실성이 증가했음을 의미합니다. 이러한 성능 저하는 과도한 클러스터 수로 인한 과적합(overfitting)에서 기인한 것으로 판단됩니다.

두 GMM 설정 간의 BIC 차이가 높지 않기 때문에, 본 모델 비교에서는 RMSE와 entropy 수준을 주요 성능 지표로 삼아 비교하였습니다. 그 결과 스트레스 조건 하에서의 최종 모델로 K = 3을 선택하였습니다.

이제 이 스트레스 테스트에서 KMeans 모델에 비해 GMM (K = 3) 모델의 전체 성능 향상을 계산합니다.

RMSE 개선 =  $\frac{RMSE_{K-means} - RMSE_{GMM}}{RMSE_{K-means}} = 29.468950777989367%$

- **최적 GMM 클러스터: 3 (시드 = 1)**
- **RMSE: 2.49**
- **기준선 대비 RMSE 개선: 29.47%**

달성된 RMSE 점수는 GMM (K = 3) 모델이 가혹한 실제 데이터 환경에서도 유저 평점을 정확하게 예측하여 유저 선호도와 잘 일치하는 추천을 제공함을 나타냅니다.

## 향후 개선 사항
- **하이브리드 추천 접근법**: 추천 정확도를 향상시키기 위해 협업 필터링과 콘텐츠 기반 필터링을 결합합니다.
- **모델 개선**: 보다 정교한 견고성 기술(특히 노이즈 / 데이터 손상에 대한 견고함) 도입 또는 신경망 추천 시스템 적용.
