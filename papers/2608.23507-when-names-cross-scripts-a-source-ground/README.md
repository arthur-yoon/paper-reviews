# When Names Cross Scripts: A Source-Grounded Benchmark for Historical Entity Reconciliation in the Mongol World

## 핵심 아이디어 한 줄 요약
역사적 인물 식별은 단순한 문자열 일치나 음역(발음 전사) 문제가 아니며, 각 증거가 어디에서 나왔는지(출처/Provenance)를 통제하여 명칭과 역사적 맥락이 어떻게 결합되어야 하는지 평가하는 벤치마크(MHER)를 구축했다.

## 구현 설명
이 논문의 핵심을 검증하는 가장 단순한 개념적 구현은 **'출처 기반 엔티티 매칭 시뮬레이터'**로 정의할 수 있다. 실제 복잡한 자연어 처리 모델 대신, 두 인물(A와 B)의 이름과 해당 이름이 등장한 '출처 문장(Proof)'을 입력으로 받아 동일인물인지 판단하는 로직을 설계하는 것이다.

구체적인 구현 단계는 다음과 같다.

1.  **데이터 구조 정의**:
    *   `PersonRecord` 객체를 정의한다. 이 객체에는 `id`, `name_variants` (예: "Tamerlane", "Timur"), 그리고 `source_evidence` (이 이름이 등장한 역사적 기록의 일부 텍스트) 리스트를 포함해야 한다.
    *   핵심은 `name_variants`만 있는 경우와, `source_evidence`가 함께 있는 경우를 분리할 수 있어야 한다는 점이다.

2.  **매칭 로직 (Naive Baseline vs. Source-Grounded)**:
    *   **Name-only 모드**: 두 `PersonRecord`의 `name_variants` 리스트를 비교한다. 편집 거리(Levenshtein distance)가 일정 임계값 이하이거나, 사전에 정의된 음역 변환 테이블(예: 'Timur' -> 'Tamerlane')에 일치하는 항목이 있으면 'Same Person'으로 판정한다.
    *   **Source-grounded 모드**: 이름 비교 외에, 두 레코드의 `source_evidence` 텍스트를 분석한다. 여기서 중요한 것은 '모순되는 정보'를 찾는 것이다. 예를 들어, A의 증거에는 "1405년 사망"이라 적혀 있고 B의 증거에는 "1405년 생"이라 적혀 있다면, 이름이 아무리 비슷해도 'Different Person'으로 판정하도록 로직을 구성한다. 반대로, 증거가 서로를 지지하거나 모순이 없다면 이름 유사도를 기반으로 판단한다.

3.  **평가 시나리오 구성**:
    *   **Case 1 (일반적 매칭)**: 서로 다른 표기지만 동일한 인물을 가리키는 쌍을 만들어, Source-grounded 로직이 이름의 차이를 극복하고 동일성을 판정하는지 확인한다.
    *   **Case 2 (동일명 다른 인물 - Homonyms)**: "Mongke"라는 이름을 가진 서로 다른 두 인물(예: 몽케 칸과 후대의 동명 인물)을 만든다. Name-only 로직은 이를 'Same'으로 오인하겠지만, Source-grounded 로직은 출처에 적힌 '생몰연도'나 '활동 지역'의 충돌을 감지하여 'Different'로 판정해야 한다.
    *   **Case 3 (거절/Abstention)**: 증거가 부족하거나 모순이 심하여 판단할 수 없는 경우, 'Unknown' 또는 'Abstain'으로 반환하도록 한다. 논문에서 언급된 'abstention' 개념을 구현하기 위함이다.

이 구현은 실제 AI 모델이 아니라 규칙 기반(rule-based) 또는 간단한 벡터 유사도 계산으로 이루어져도, **"출처 증거가 없으면 이름만으로는 판단이 불가능하다"**는 논문의 주장을 구조적으로 재현할 수 있다.

## 실행 방법과 예상 출력
상기 시뮬레이터를 실행할 때는, MHER 벤치마크에 포함된 대표적인 케이스들을 샘플링하여 입력으로 주어야 한다.

**실행 시나리오:**
1.  **입력 데이터 로드**: `MHER_NameOnly.csv`와 `MHER_SourceGrounded.csv` 파일의 일부 행을 추출한다.
2.  **테스트 케이스 1: 음역이 다른 동일 인물**
    *   입력: Person A (Name: "Genghis Khan", Source: "Chinghis Khan of the Steppe"), Person B (Name: "Temujin", Source: "Temujin, founder of the empire")
    *   실행: `match_persons(A, B, mode="source_grounded")` 호출
    *   **예상 출력**: `Decision: SAME`. Reason: "Source evidence indicates historical context aligns (founder/steppe), overcoming name surface mismatch." (이름은 다르지만 출처가 같은 역사적 맥락을 가리키므로 동일인으로 판정)
3.  **테스트 케이스 2: 이름이 완전히 같은 다른 인물 (Critical Failure Case)**
    *   입력: Person A (Name: "Batu", Source: "Batu, Khan of the Golden Horde, died 1255"), Person B (Name: "Batu", Source: "Batu, a minor official in 1300s")
    *   실행: `match_persons(A, B, mode="name_only")` 호출 -> **예상 출력**: `Decision: SAME` (오류 발생. 이름이 같아서)
    *   실행: `match_persons(A, B, mode="source_grounded")` 호출 -> **예상 출력**: `Decision: DIFFERENT`. Reason: "Source evidence contradicts temporal context (1255 vs 1300s)."
4.  **테스트 케이스 3: 불충분한 증거**
    *   입력: Person A (Name: "Qutlugh", Source: "Qutlugh, merchant"), Person B (Name: "Qutlugh", Source: "Qutlugh, soldier")
    *   실행: `match_persons(A, B, mode="source_grounded")` 호출
    *   **예상 출력**: `Decision: ABSTAIN`. Reason: "Insufficient provenance to resolve identity; names identical but sources do not provide linking or explicitly contradicting high-level identity markers."

이 실행을 통해, Name-only 모델이 Case 2에서 발생하는 치명적인 오류를 Source-grounded 접근법이 어떻게 수정하는지를 시각적으로 확인할 수 있다. 특히, 논문에서 지적한 것처럼 단순한 이름 매칭이 실패하는 지점(동일명 다른 인물)에서 출처 정보가 결정적인 역할을 하는 것을 입증할 수 있다.

## 한계
이러한 간이 구현은 실제 논문이 제안한 MHER 벤치마크와 생성형 AI 모델 평가 방식과 몇 가지 근본적인 차이가 있다.

1.  **추론 능력의 부재**: 위의 구현은 규칙 기반이거나 단순 텍스트 매칭에 의존한다. 실제 논문에서는 Qwen3-8B 같은 생성형 LLM이 복잡한 역사적 문맥을 '이해'하고 추론하는 과정을 평가한다. LLM은 명시적으로 언급되지 않은 암묵적 단서(예: 특정 관직의 시대를 이용한 연대 확정)를 활용할 수 있지만, 규칙 기반 시뮬레이터는 이러한 추론을 수행할 수 없다.
2.  **'Context-only' 아블레이션의 복잡성**: 논문에서는 '이름 없이 문맥만' 제공할 경우에도 상당한 정확도가 나오지만, 이름이 돌아오면 오히려 오류가 증가할 수 있음을 보여준다(Qwen3-8B 사례). 이는 모델 내부의 '이름 고정(anchoring) 편향'을 시사한다. 간이 구현에서는 이러한 모델의 인지적 편향이나 학습된 사전 지식과의 충돌을 재현하기 어렵다.
3.  **벤치마크 규모와 균형**: MHER는 84명의 주체 인물, 396쌍의 Name-only, 160쌍의 Source-grounded 데이터로 구성되어 있으며, 개발/테스트 세트가 엔티티 단위로 분리되어 데이터 누출을 막는다. 간이 구현은 소수의 수동 정의된 케이스를 다루므로, 통계적 유의성이나 다양한 역사적 시나리오를 포괄하는 일반화 가능성이 낮다.
4.  **거절(Abstention)의 질적 평가**: 논문은 모델이 '잘못 추측하는 것'보다 '모르다고 말하는 것'을 얼마나 잘 수행하는지(거절 능력)를 중요한 평가 지표로 삼는다. 간이 구현은 '정보 부족'을 판단하는 명확한 기준을 설정하기 어렵고, 실제 AI 모델이 보이는 모호한 확도(confidence)나 추론 과정(chain-of-thought)의 질을 평가하는 것이 불가능하다.

따라서 이 구현은 '출처 프로브네이션(Provenance)의 중요성'이라는 개념을 논리적으로 검증하기 위한 교육용 또는 프로토타입 용도로만 사용 가능하며, 실제 역사적 엔티티 리졸루션 시스템의 성능 평가나 AI 모델의 역사적 이해력을 측정하는 데는 사용될 수 없다.


---
*생성: qwen3.8-27b (qwen-local)*
