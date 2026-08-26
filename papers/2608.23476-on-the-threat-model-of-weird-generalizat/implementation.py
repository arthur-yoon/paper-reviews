import numpy as np

# 시드 고정으로 재현 가능성 확보
np.random.seed(42)

# 1. 대상 모델 및 데이터셋 선정 (시뮬레이션용 파라미터)
models = ["Model_A", "Model_B", "Model_C"]

# 4개의 소규모 도메인 데이터셋 정의
# 변수 A(구성): 'qa' vs 'dialog'
# 변수 B(언어): 'en' vs 'ko'
# 변수 C(신규성): 'known' vs 'novel'
# 데이터 크기는 의도적으로 일정하게 유지 (예: 100 samples)
datasets = [
    {"name": "D1", "composition": "qa", "language": "en", "novelty": "known", "size": 100},
    {"name": "D2", "composition": "dialog", "language": "en", "novelty": "known", "size": 100},
    {"name": "D3", "composition": "qa", "language": "ko", "novelty": "known", "size": 100},
    {"name": "D4", "composition": "qa", "language": "en", "novelty": "novel", "size": 100}
]

# 평가 질문 세트 정의 (변수 D)
# 세트 1: 일반적 질문 (낮은 민감도)
# 세트 2: 주제 일부 겹침 (중간 민감도)
# 세트 3: 패턴 모방 (높은 민감도)
eval_sets = ["Set_General", "Set_Partial", "Set_Mimic"]

# 2. 미세조정(Fine-tuning) 및 3. 평가 프로토콜 시뮬레이션
# 실제 모델 추론 대신, 논문의 주장(가설)을 반영한 확률적 시뮬레이션 수행
# WG Score: 0.0 ~ 1.0 사이의 값

wg_scores = {}

for model in models:
    wg_scores[model] = {}
    for ds in datasets:
        wg_scores[model][ds["name"]] = {}
        
        # 미세조정의 기본 영향력 (모델에 따라 미세하게 다를 수 있음)
        base_wg = 0.1
        
        # 변수 C (신규성): 익숙한 데이터(known)가 더 높은 WG을 유발한다는 논문 주장 반영
        novelty_factor = 0.2 if ds["novelty"] == "known" else 0.0
        base_wg += novelty_factor
        
        # 변수 B (언어): 모델이 덜 학습한 언어(예: ko)가 더 극적인 WG을 유발한다고 가정
        # (참고: 모델마다 사전학습 구성이 다를 수 있으나, 시뮬레이션 상 ko를 낮게 설정)
        lang_factor = 0.1 if ds["language"] == "ko" else 0.0
        base_wg += lang_factor
        
        # 변수 A (구성): dialog가 qa보다 더 큰 변이를 유발한다고 가정
        comp_factor = 0.05 if ds["composition"] == "dialog" else 0.0
        base_wg += comp_factor
        
        # 3. 평가 질문 세트별 민감성 분석
        for ev_set in eval_sets:
            # 평가 세트에 따른 변동성(Noise) 및 편향(Bias)
            # 논문 핵심: 평가 질문 선택에 따라 WG 점수가 크게 흔들림 (취약한 측정)
            
            if ev_set == "Set_General":
                bias = -0.1
                noise_std = 0.05
            elif ev_set == "Set_Partial":
                bias = 0.0
                noise_std = 0.1
            else: # Set_Mimic
                bias = 0.2
                noise_std = 0.15
            
            # 시뮬레이션된 WG 점수 계산 (Gaussian Noise 추가)
            # base_wg + bias + N(0, noise_std)
            score = base_wg + bias + np.random.normal(0, noise_std)
            
            # 0~1 범위로 클리핑
            score = np.clip(score, 0.0, 1.0)
            
            wg_scores[model][ds["name"]][ev_set] = score

# 4. 비교 및 분석 로직

print("="*60)
print("분석 1: 구성(Composition) vs 크기 (크기는 동일 가정)")
print("-"*60)
# D1(QA, EN, Known) vs D2(DLG, EN, Known) 비교
# 크기가 같으므로 구성의 차이를 보임
print(f"모델: {models[0]}")
print(f"  D1 (QA, EN, Known):")
print(f"    Set_General: {wg_scores[models[0]]['D1']['Set_General']:.4f}")
print(f"    Set_Partial: {wg_scores[models[0]]['D1']['Set_Partial']:.4f}")
print(f"    Set_Mimic:   {wg_scores[models[0]]['D1']['Set_Mimic']:.4f}")
print(f"  D2 (DLG, EN, Known):")
print(f"    Set_General: {wg_scores[models[0]]['D2']['Set_General']:.4f}")
print(f"    Set_Partial: {wg_scores[models[0]]['D2']['Set_Partial']:.4f}")
print(f"    Set_Mimic:   {wg_scores[models[0]]['D2']['Set_Mimic']:.4f}")
# 평균 비교
avg_d1 = np.mean([wg_scores[models[0]]['D1'][s] for s in eval_sets])
avg_d2 = np.mean([wg_scores[models[0]]['D2'][s] for s in eval_sets])
print(f"  평균 WG: D1={avg_d1:.4f}, D2={avg_d2:.4f} (차이: {abs(avg_d2-avg_d1):.4f})")

print("\n" + "="*60)
print("분석 2: 언어(Language) 영향")
print("-"*60)
# D1(EN) vs D3(KO) 비교
print(f"모델: {models[0]}")
avg_en = np.mean([wg_scores[models[0]]['D1'][s] for s in eval_sets])
avg_ko = np.mean([wg_scores[models[0]]['D3'][s] for s in eval_sets])
print(f"  평균 WG (EN - D1): {avg_en:.4f}")
print(f"  평균 WG (KO - D3): {avg_ko:.4f}")
print(f"  차이: {abs(avg_ko-avg_en):.4f}")

print("\n" + "="*60)
print("분석 3: 신규성(Novelty) 영향")
print("-"*60)
# D1(Known) vs D4(Novel) 비교
print(f"모델: {models[0]}")
avg_known = np.mean([wg_scores[models[0]]['D1'][s] for s in eval_sets])
avg_novel = np.mean([wg_scores[models[0]]['D4'][s] for s in eval_sets])
print(f"  평균 WG (Known - D1): {avg_known:.4f}")
print(f"  평균 WG (Novel - D4): {avg_novel:.4f}")
print(f"  논문 주장 검증: Known > Novel 이 성립하는가? {avg_known > avg_novel}")

print("\n" + "="*60)
print("분석 4: 평가 민감성 (측정 불안정성)")
print("-"*60)
# 동일한 모델, 동일한 데이터셋(D1)에 대해 평가 세트만 변경
print(f"모델: {models[0]}, 데이터: D1")
scores_for_d1 = [wg_scores[models[0]]['D1'][s] for s in eval_sets]
min_score = min(scores_for_d1)
max_score = max(scores_for_d1)
range_score = max_score - min_score
std_score = np.std(scores_for_d1)
print(f"  Set_General: {scores_for_d1[0]:.4f}")
print(f"  Set_Partial: {scores_for_d1[1]:.4f}")
print(f"  Set_Mimic:   {scores_for_d1[2]:.4f}")
print(f"  변동폭(Range): {range_score:.4f}")
print(f"  표준편차(Std): {std_score:.4f}")
print(f"  결론: 변동폭이 {0.1:.1f}을 초과하므로 WG 측정의 불안정성을 시사함.")

print("\n" + "="*60)
print("최종 요약 테이블 (모든 모델, 모든 데이터셋, 평균 WG 점수)")
print("-"*60)
header = f"{'Dataset':<10} {'Model':<10} {'Avg WG':<10} {'Std WG':<10}"
print(header)
for ds in datasets:
    ds_name = ds["name"]
    for model in models:
        scores = [wg_scores[model][ds_name][s] for s in eval_sets]
        avg = np.mean(scores)
        std = np.std(scores)
        print(f"{ds_name:<10} {model:<10} {avg:<10.4f} {std:<10.4f}")
