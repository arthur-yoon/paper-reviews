# Paper Reviews & Implementations

AI/ML 논문 리뷰와, 논문의 핵심 아이디어를 실제로 검증 가능한 형태로 구현한 저장소입니다.

- 각 논문 폴더는 **독립된 `uv` 프로젝트**입니다 — 클론 후 아래처럼 바로 실행됩니다.
- 구현은 논문의 핵심 아이디어를 이해하기 위한 **간이(simplified) 구현**이며, 실제 논문의
  전체 실험을 재현하지는 않습니다(각 폴더 README의 "한계" 절 참고). 다만 코드는 실제로
  실행되고, `uv run pytest`로 검증된 테스트를 통과합니다 — 받자마자 바로 돌려볼 수
  있는 것을 목표로 합니다.
- 자동 생성 파이프라인: content-engine (Qwen3.8-27B, 로컬 실행 + 실제 실행 검증)

```bash
cd papers/<논문 폴더>
uv sync                 # 의존성 설치
uv run pytest -q        # 테스트 실행
uv run jupyter notebook demo.ipynb   # (있는 경우) 결과 시각화
```

## 구조

```
papers/
  <arxiv-id>-<slug>/
    pyproject.toml         # uv 프로젝트 정의
    README.md              # 원본 논문 링크 + 상세 리뷰 + 논문<->코드 매핑
    implementation.py      # 구현 (로깅, 타입힌트, docstring 포함)
    test_implementation.py # 실행 검증된 테스트
    demo.ipynb   # (있는 경우) 결과 시각화 노트북
```

현재 15건.

## 전체 목록

| 논문 | 태그 | uv 프로젝트 | 테스트 | 노트북 |
|---|---|---|---|---|
| [On the Threat Model of Weird Generalization and Emergent Misalignment](papers/2608.23476-on-the-threat-model-of-weird-generalizat/) | - | - | - | - |
| [Mitigating Reasoning-Induced Misalignment via Safety-Direction Penalty](papers/2608.23497-mitigating-reasoning-induced-misalignmen/) | - | - | - | - |
| [When Names Cross Scripts: A Source-Grounded Benchmark for Historical Entity Reconciliation in the Mongol World](papers/2608.23507-when-names-cross-scripts-a-source-ground/) | - | - | - | - |
| [How to Train a Critic Stably and Efficiently](papers/2608.23566-how-to-train-a-critic-stably-and-efficie/) | - | - | - | - |
| [Method, Mind, and Morality: How People Make Sense of Artificial Intelligence](papers/2608.24748-method-mind-and-morality-how-people-make/) | - | ✅ | ✅ | - |
| [The RAT: A Unified Bayesian Model for RAG Evaluation](papers/2608.24753-the-rat-a-unified-bayesian-model-for-rag/) | - | ✅ | ✅ | ✅ |
| [A Geometric Theory of Robust Fairness Audits](papers/2608.24818-a-geometric-theory-of-robust-fairness-au/) | - | ✅ | ✅ | ✅ |
| [Constrained Entity Selection under Partial Knowledge for LLM-Based Knowledge Graph QA](papers/2608.24824-constrained-entity-selection-under-parti/) | - | - | - | - |
| [A Dual-Dimensional LLM Framework for Automated Item Incidental Content Similarity Analysis in Large-Scale Assessments](papers/2608.24825-a-dual-dimensional-llm-framework-for-aut/) | - | - | - | - |
| [Reading Is Not Using: Retrieval, Judgment, and the Design of AI Financial Research Workflows](papers/2608.24842-reading-is-not-using-retrieval-judgment/) | - | - | - | - |
| [FedV-KGQA: Multi-Hop Question Answering over Vertically Partitioned Knowledge Graphs](papers/2608.24846-fedv-kgqa-multi-hop-question-answering-o/) | - | - | - | - |
| [Bellman Calibration for Marginalized Importance Weighting in Offline Reinforcement Learning](papers/2608.24858-bellman-calibration-for-marginalized-imp/) | - | - | - | - |
| [Improving Cross-Problem Vehicle Routing with Locally Augmented Preferences and Representation Disentanglement](papers/2608.24859-improving-cross-problem-vehicle-routing/) | - | - | - | - |
| [Parameterized Complexity of $L_p$-Lipschitz Constants for Input Convex Neural Networks and $L_p$-Norm Maximization over Zonotopes](papers/2608.24865-parameterized-complexity-of-l-p-lipschit/) | - | - | - | - |
| [What FID Hides: Detecting, Ranking, and Diagnosing Deviations in Generative Evaluation](papers/2608.24881-what-fid-hides-detecting-ranking-and-dia/) | - | - | - | - |

## 주제별 분류

### 미분류

- [On the Threat Model of Weird Generalization and Emergent Misalignment](papers/2608.23476-on-the-threat-model-of-weird-generalizat/)
- [Mitigating Reasoning-Induced Misalignment via Safety-Direction Penalty](papers/2608.23497-mitigating-reasoning-induced-misalignmen/)
- [When Names Cross Scripts: A Source-Grounded Benchmark for Historical Entity Reconciliation in the Mongol World](papers/2608.23507-when-names-cross-scripts-a-source-ground/)
- [How to Train a Critic Stably and Efficiently](papers/2608.23566-how-to-train-a-critic-stably-and-efficie/)
- [Method, Mind, and Morality: How People Make Sense of Artificial Intelligence](papers/2608.24748-method-mind-and-morality-how-people-make/)
- [The RAT: A Unified Bayesian Model for RAG Evaluation](papers/2608.24753-the-rat-a-unified-bayesian-model-for-rag/)
- [A Geometric Theory of Robust Fairness Audits](papers/2608.24818-a-geometric-theory-of-robust-fairness-au/)
- [Constrained Entity Selection under Partial Knowledge for LLM-Based Knowledge Graph QA](papers/2608.24824-constrained-entity-selection-under-parti/)
- [A Dual-Dimensional LLM Framework for Automated Item Incidental Content Similarity Analysis in Large-Scale Assessments](papers/2608.24825-a-dual-dimensional-llm-framework-for-aut/)
- [Reading Is Not Using: Retrieval, Judgment, and the Design of AI Financial Research Workflows](papers/2608.24842-reading-is-not-using-retrieval-judgment/)
- [FedV-KGQA: Multi-Hop Question Answering over Vertically Partitioned Knowledge Graphs](papers/2608.24846-fedv-kgqa-multi-hop-question-answering-o/)
- [Bellman Calibration for Marginalized Importance Weighting in Offline Reinforcement Learning](papers/2608.24858-bellman-calibration-for-marginalized-imp/)
- [Improving Cross-Problem Vehicle Routing with Locally Augmented Preferences and Representation Disentanglement](papers/2608.24859-improving-cross-problem-vehicle-routing/)
- [Parameterized Complexity of $L_p$-Lipschitz Constants for Input Convex Neural Networks and $L_p$-Norm Maximization over Zonotopes](papers/2608.24865-parameterized-complexity-of-l-p-lipschit/)
- [What FID Hides: Detecting, Ranking, and Diagnosing Deviations in Generative Evaluation](papers/2608.24881-what-fid-hides-detecting-ranking-and-dia/)

