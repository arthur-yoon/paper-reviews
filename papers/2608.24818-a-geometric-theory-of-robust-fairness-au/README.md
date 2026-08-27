# A Geometric Theory of Robust Fairness Audits

## 원본 논문

**A Geometric Theory of Robust Fairness Audits** — [arXiv 원문](http://arxiv.org/abs/2608.24818v1)

## 빠른 시작

```bash
uv sync
uv run pytest -q               # 테스트 실행
uv run jupyter notebook notebooks/demo.ipynb   # 결과 시각화 노트북 열기
```

## 논문 핵심 내용

이 논문은 개별 공정성(individual fairness)을 검증하는 '근처 기반 공정성 감사(neighborhood-based fairness audit)' 방법론의 **견고함(robustness)**을 기하학적 관점에서 분석하는 데 초점을 맞춘 연구입니다. 현재 많은 연구와 실무에서 모델의 공정성을 평가할 때, 특징 공간(feature space) 내에서 서로 유사한 개인들의 예측값이 크게 다르지 않다는 기준을 사용합니다. 즉, 가까운 이웃(neighbors)끼리 예측 결과가 일관되어야 공정한 모델이라고 판단하는 것이죠. 하지만 이 논문은 이러한 감사 절차 자체에 심각한 취약성이 존재한다고 지적합니다. 특징 공간의 아주 작은 섭동(perturbation)만으로도 특정 데이터 포인트의 '가장 가까운 이웃' 관계가 뒤바뀌고, 그 결과 모델의 예측값이 그대로인데도 불구하고 공정성 평가 결과가 완전히 달라질 수 있다는 문제를 정의합니다.

기존의 접근법은 주로 모델 출력의 민감도나 데이터의 대표성 문제를 다뤘지만, '감사 절차(audit procedure) 자체의 기하학적 불안정성'을 정량적으로 분석한 연구는 드물었습니다. 기존 한계는 이웃 관계가 비연속적(discontinuous)이라는 점을 무시하고, 이웃이 고정된 것처럼 가정했다는 데 있습니다. 만약 데이터 포인트 $x$의 k-최근접 이웃 집합이 $N_k(x)$라면, $x$가 아주 살짝 이동해도 $N_k(x)$가 완전히 다른 집합으로 교체(replacement)될 수 있으며, 이는 감사 지표(audit metric)의 급격한 변화를 유발합니다.

이 논문의 핵심 기여는 이러한 현상을 이해하기 위한 **기하학적 프레임워크**를 구축한 것입니다. 연구자들은 유계 섭동(bounded perturbations) 하에서 이웃이 변하지 않는 '이웃 불변성(neighborhood invariance)'을 보장하기 위한 충분 조건을 수학적으로 증명했습니다. 또한, 이웃이 교체되었을 때 그 영향이 공정성 감사 결과의 불안정성으로 어떻게 전파(propagate)되는지를 정량화했습니다. 특히 주목할 점은 새로운 측정 지표인 **'감사 변동성(audit volatility)'**을 제안했다는 것입니다. 이 지표는 반복적인 섭동 적용 시 공정성 감사 결과가 얼마나 민감하게 변하는지를 기대값(expectation)으로 측정합니다. 이는 단순한 최대 오차가 아니라, 전반적인 안정성의 통계적 특성을 포착하려는 시도로 볼 수 있습니다.

이 연구의 중요성은 실무적 차원에서 매우 큽니다. 공정성 감사는 AI 시스템의 신뢰성을 검증하는 핵심 절차이므로, 이 절차 자체가 노이즈에 너무 민감하다면 감사 결과를 신뢰할 수 없습니다. 본 논문은 벤치마크 데이터셋을 통한 실험으로 이론적 분석의 타당성을 뒷받침하며, 제안한 프레임워크가 관측되는 감사 안정성을 잘 설명함을 입증합니다. 이는 데이터 전처리 단계에서의 노이즈 관리나 이웃 선정 알고리즘의 파라미터(k값) 설정에 과학적인 근거를 제공하며, '안정적인 공정성 감사'를 위한 새로운 설계 원칙을 제시한다는 점에서 학술적 및 실무적 가치를 가집니다.

## 구현 설명

논문의 핵심 개념인 '이웃 불변성'과 '감사 변동성'을 Python과 scikit-learn, NumPy를 사용하여 단순화하여 구현하는 전략은 다음과 같습니다.

1.  **데이터 생성 및 전처리**:
    *   `make_classification`이나 `make_blobs`를 사용하여 2D 또는 3D 특징 공간에 데이터 포인트를 생성합니다.
    *   각 데이터 포인트에 대한 '참값(ground truth)' 또는 '모델 예측값(y_pred)'을 생성합니다. 여기서는 실제 모델 대신, 특정 함수(예: 선형 회귀 또는 의사 결정 트리)를 사용하여 예측값을 생성하거나, 랜덤한 라벨을 부여해 테스트할 수 있습니다.

2.  **근처 기반 공정성 감사 함수 (`calculate_audit_score`)**:
    *   입력: 데이터 포인트 X, 예측값 y, k(이웃 개수).
    *   로직: `KNeighborsRegressor` 또는 `NearestNeighbors`를 이용해 각 샘플의 k-최근접 이웃을 찾습니다.
    *   계산: 각 샘플에 대해, 자기 자신의 예측값과 k-최근접 이웃들의 예측값 사이의 평균 절대 오차(MAE)나 표준편차를 계산합니다.
    *   출력: 전체 데이터셋에 대한 평균 공정성 점수(값이 낮을수록 공정)를 반환합니다.

3.  **섭동(Simulation of Perturbation) 함수 (`apply_perturbation`)**:
    *   입력: 원본 데이터 X, 섭동 크기(epsilon).
    *   로직: X의 각 특징 값에 Uniform(-epsilon, epsilon) 분포를 따르는 랜덤 노이즈를 더합니다.
    *   출력: 섭동이 적용된 새로운 데이터셋 $X'$.

4.  **이웃 불변성 검사 (`check_neighborhood_invariance`)**:
    *   입력: 원본 X, 섭동된 X', k.
    *   로직: 원본 X에서 각 포인트의 k-최근접 이웃 집합 ID 목록을 추출합니다. 섭동된 X'에서도 동일하게 추출합니다.
    *   비교: 두 목록의 Jaccard Similarity(재키드 유사도)를 계산합니다. 1.0에 가까울수록 이웃 관계가 불변(invariant)합니다.
    *   출력: 평균 Jaccard Similarity 또는 이웃이 완전히 교체된 포인트의 비율.

5.  **감사 변동성(Audit Volatility) 계산 (`calculate_audit_volatility`)**:
    *   입력: X, y, epsilon, trial_count(반복 횟수).
    *   로직:
        1.  `trial_count`번 반복합니다.
        2.  매 반복마다 `apply_perturbation`으로 새로운 $X_t$ 생성.
        3.  $X_t$를 기반으로 `calculate_audit_score` 호출하여 공정성 점수 $S_t$ 계산.
        4.  모든 $S_t$의 표준편차(Standard Deviation)나 분산(Variance)을 계산합니다.
    *   출력: 계산된 표준편차를 'Audit Volatility'로 반환합니다. 이 값이 작을수록 감사 절차가 견고합니다.

이 구현은 복잡한 기하학적 증명 대신, 수치적 시뮬레이션을 통해 논문의 주장(섭동이 크면 변동성이 커진다)을 경험적으로 검증하는 데 초점을 둡니다.

## 논문 ↔ 코드 매핑

| 논문 부분 (개념/섹션) | 구현 위치 (함수/클래스) | 비고 |
|---|---|---|
| Section: Neighborhood-based Fairness Definition | `calculate_audit_score(X, y, k)` | k-최근접 이웃의 예측값 차이로 공정성을 측정하는 로직. 논문에서 정의한 audit metric의 구체적 실현. |
| Section: Bounded Perturbations | `apply_perturbation(X, epsilon)` | 유계 섭동 모델을 구현. Uniform 노이즈 추가를 통해 bounded condition 충족. |
| Section: Neighborhood Invariance Conditions | `check_neighborhood_invariance(X, X_perturbed, k)` | 이웃 집합의 변화 정도를 Jaccard Similarity로 측정. 논문의 'invariance' 개념을 수치적으로 검증하는 함수. |
| Section: Audit Volatility Definition | `calculate_audit_volatility(X, y, epsilon, trials)` | 반복 시뮬레이션을 통해 공정성 점수의 표준편차 계산. 논문의 핵심 기여 지표인 'volatility' 구현. |
| Section: Experimental Setup | `generate_synthetic_data(n_samples, dim)` | 실험을 위해 필요한 벤치마크 데이터 생성 로직. |
| Section: Main Results / Stability Analysis | `main()` or `run_experiments()` | epsilon(섭동 크기)을 다양하게 변화시키며 `calculate_audit_volatility`를 호출하고, epsilon과 volatility의 상관관계를 시각화/출력하는 메인 루프. |

## 실행 방법과 예상 출력

**실행 방법:**

1.  `python robust_fairness_audit.py`를 실행합니다.
2.  스크립트는 다음과 같은 단계를 따릅니다:
    *   2D 특징 공간에 1000개의 데이터 포인트와 임의의 예측값을 생성합니다.
    *   `k=5`로 설정한 근접 기반 공정성 감사를 수행합니다.
    *   섭동 크기(epsilon)를 0.0, 0.1, 0.5, 1.0으로 단계별로 증가시킵니다.
    *   각 epsilon 값에 대해 100회 반복 시뮬레이션하여 'Audit Volatility'(표준편차)와 '이웃 불변성'(평균 Jaccard Similarity)을 계산합니다.
    *   결과를 테이블 형태로 출력하고, epsilon에 따른 변동성 변화를 그래프(matplotlib)로 그립니다.

**예상 출력:**

```text
Generating synthetic data: 1000 samples in 2D space.
Base Audit Score (no perturbation): 0.1245

Starting Robustness Analysis...

Epsilon | Avg Neighborhood Invariance (Jaccard) | Audit Volatility (Std Dev)
--------------------------------------------------------------------------------
0.00    | 1.0000                                | 0.0000
0.10    | 0.8532                                | 0.0153
0.50    | 0.4210                                | 0.0892
1.00    | 0.1125                                | 0.2145

Analysis Complete.
Observation: As epsilon increases, neighborhood invariance drops significantly,
leading to a substantial increase in audit volatility.
This confirms that neighborhood-based fairness audits are sensitive to feature space perturbations.
Saved plot: volatility_vs_epsilon.png
```

출력 결과에서 epsilon이 증가함에 따라 Jaccard Similarity(이웃의 일관성)가 급격히 하락하고, 동시에 Audit Volatility(감사 결과의 흔들림)가 커지는 것을 확인할 수 있습니다. 이는 논문에서 주장한 "이웃 관계의 불안정성이 공정성 감사의 불안정성으로 전파된다"는 내용을 수치적으로 증명해 줍니다.

## 한계

1.  **이론적 추상화와의 격차**: 본 구현은 논문의 기하학적 '충분 조건'을 직접적으로 검증하는 것이 아니라, 수치적 시뮬레이션을 통해 현상을 관찰하는 수준입니다. 고차원 데이터에서 발생하는 '차원의 저주' 효과나, 특정한 기하학적 구조(예: 데이터가 초평면에 집중된 경우)에 대한 이론적 보장은 이 간단한 예시에서는 확인되지 않습니다.
2.  **이웃 선정 알고리즘의 단순성**: 실제 생산 환경에서는 `KNN` 외에도 DBSCAN, HDBSCAN 등 다양한 밀도 기반 클러스터링 알고리즘이 사용될 수 있습니다. 본 구현은 k-최근접 이웃에 한정되므로, 다른 이웃 정의 방식에서의 견고함은 알 수 없습니다.
3.  **모델 예측값의 정합성 가정**: 실제 공정성 감사에서는 모델의 예측값이 학습 데이터에 의해 결정되므로, 특징 공간의 섭동이 예측값에도 영향을 미칠 수 있습니다. 본 예시는 '모델 예측값이 고정되어 있음'을 전제하고 감사 절차 자체의 기하학적 불안정성만 분리해 보는 것입니다. 이는 논문의 특정 관점(감사 절차의 노이즈 민감도)에 맞지만, 전체적인 시스템의 공정성 변화(모델 출력 변화 포함)를 반영하지는 못합니다.
4.  **계산 비용**: `calculate_audit_volatility` 함수는 반복 시뮬레이션을 수행하므로, 데이터 포인트 수가 많거나 반복 횟수가 크면 계산 비용이 기하급수적으로 증가합니다. 대규모 프로덕션 데이터셋에 바로 적용하기 위해서는 샘플링 기반 근사 방법이나 몬테카를로 시뮬레이션의 최적화가 필요합니다.


---
*생성: qwen3.8-27b (qwen-local)*
