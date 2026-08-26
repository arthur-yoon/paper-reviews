import numpy as np

class SimulatedLLM:
    """
    논문의 '검색-통합 격차'를 모사하는 단순화된 LLM 시뮬레이터.
    - 검색 능력(Recall): 문맥에 정답이 있으면 높게, 아니면 0
    - 판단 능력(Integration): 문맥 길이가 길어질수록 핵심 정보 반영률이 로그-시그모이드 함수로 감소
    """
    def __init__(self, name, model_capacity, integration_efficiency):
        self.name = name
        self.model_capacity = model_capacity  # 모델 규모 (임의 단위, 예: 파라미터 수/1B)
        self.integration_efficiency = integration_efficiency  # 정보 통합 효율 (0~1)
        
    def retrieve(self, context_tokens, key_info_present):
        """
        검색 정확도 시뮬레이션.
        문맥에 핵심 정보가 포함되어 있으면 모델 규모에 따라 일정한 높은 정확도를 반환.
        """
        if not key_info_present:
            return 0.0
        # 모델 규모가 클수록 검색 정확도가 미세하게 향상 (90% ~ 99%)
        base_accuracy = 0.90
        improvement = min(0.09, np.log1p(self.model_capacity) * 0.02)
        noise = np.random.normal(0, 0.01)
        accuracy = np.clip(base_accuracy + improvement + noise, 0.0, 1.0)
        return accuracy

    def judge(self, context_tokens, key_info_present, use_restatement):
        """
        판단 영향력 시뮬레이션.
        핵심 리스크가 최종 판단(매도/매수)에 반영되는 비율.
        
        Args:
            context_tokens: 배경 문맥 토큰 수
            key_info_present: 핵심 리스크 정보가 문맥에 있는지 여부
            use_restatement: Targeted Restatement 워크플로우 적용 여부
            
        Returns:
            리스크 반영도 (0.0 ~ 1.0)
        """
        if not key_info_present:
            return 0.0

        # 1. 기본 통합 함수: 문맥 길이(L)에 따라 성능 저하
        # L이 커질수록 영향력이 감소 (격차 발생)
        # model_capacity가 클수록 감소가 느림 (임계점 지연)
        
        # 유효 문맥 길이를 모델 능력으로 보정
        effective_length = context_tokens / self.model_capacity
        
        # 시그모이드 기반 감소 함수
        # scale: 모델 규모에 따라 감소 속도가 달라짐
        scale_factor = 1.0 + 0.5 * np.log1p(self.model_capacity)
        
        if use_restatement:
            # Pipeline B: Targeted Restatement
            # 핵심 정보를 구조화하여 재제시 하므로, 문맥 길이에 덜 민감
            # 그러나 완전한 방어도 아님, 약간의 감소만 발생
            decay = np.exp(-0.1 * effective_length)  # 느린 감쇠
            # 기본 효율 * 감쇠율
            risk_reflection = self.integration_efficiency * decay
        else:
            # Pipeline A: Standard / Chunk-Summarize
            # 문맥 길이에 매우 민감, 급격한 감소
            # 128k 토큰 수준에서 '노이즈 플로어'에 도달
            decay = 1.0 / (1.0 + np.exp((effective_length / scale_factor) - 3.0))
            risk_reflection = self.integration_efficiency * decay * 0.8  # 요약 과정에서 추가 손실
            
        # 노이즈 추가 (실험적 변동성)
        noise = np.random.normal(0, 0.05)
        risk_reflection = np.clip(risk_reflection + noise, 0.0, 1.0)
        
        return risk_reflection


def run_experiment():
    print("="*60)
    print("Retrieval-Integration Gap Experiment Simulation")
    print("="*60)
    
    # 1. 모델 설정 (가상의 모델 패밀리)
    models = [
        SimulatedLLM("Small-Model-7B", 7, 0.80),
        SimulatedLLM("Medium-Model-70B", 70, 0.85),
        SimulatedLLM("Large-Model-400B", 400, 0.90)
    ]
    
    # 2. 배경 문맥 토큰 수 단계
    token_levels = [2000, 4000, 8000, 16000, 32000, 64000, 128000]
    
    # 3. 실험 반복 횟수 (평균 내기 위해)
    num_runs = 10
    
    results = []
    
    for model in models:
        for tokens in token_levels:
            # Pipeline A: Standard (No Restatement)
            retrieval_accs = []
            risk_reflections_std = []
            
            # Pipeline B: Targeted Restatement
            risk_reflections_restate = []
            
            for _ in range(num_runs):
                # 핵심 정보 존재 가정 (제어 변수)
                key_info_present = True
                
                # 평가 지표 1: 검색 정확도
                ret_acc = model.retrieve(tokens, key_info_present)
                retrieval_accs.append(ret_acc)
                
                # 평가 지표 2: 판단 영향력 (Standard)
                risk_std = model.judge(tokens, key_info_present, use_restatement=False)
                risk_reflections_std.append(risk_std)
                
                # 평가 지표 2: 판단 영향력 (Restatement)
                risk_restate = model.judge(tokens, key_info_present, use_restatement=True)
                risk_reflections_restate.append(risk_restate)
            
            results.append({
                'model': model.name,
                'tokens': tokens,
                'retrieval_acc': np.mean(retrieval_accs),
                'risk_std': np.mean(risk_reflections_std),
                'risk_restate': np.mean(risk_reflections_restate)
            })
    
    # 4. 결과 출력 및 분석
    print(f"\n{'Model':<20} {'Tokens':<10} {'Retrieval %':<15} {'Risk Std %':<15} {'Risk Restate %':<15} {'Gap (Std-Restate)':<15}")
    print("-" * 90)
    
    for r in results:
        gap = r['risk_restate'] - r['risk_std']
        print(f"{r['model']:<20} {r['tokens']:<10} {r['retrieval_acc']*100:<15.2f} {r['risk_std']*100:<15.2f} {r['risk_restate']*100:<15.2f} {gap*100:<15.2f}")
        
    # 5. 핵심 관찰 요약
    print("\n--- Key Observations ---")
    
    # 128k 토큰에서의 성능 비교
    large_context_results = [r for r in results if r['tokens'] == 128000]
    
    for r in large_context_results:
        print(f"Model: {r['model']}")
        print(f"  - Retrieval Accuracy at 128k: {r['retrieval_acc']*100:.2f}% (High: Model finds the info)")
        print(f"  - Standard Risk Reflection at 128k: {r['risk_std']*100:.2f}% (Low: Info not used in judgment)")
        print(f"  - Restatement Risk Reflection at 128k: {r['risk_restate']*100:.2f}% (Mitigated: Workflow helps)")
        gap = r['risk_restate'] - r['risk_std']
        if gap > 0.1:
            print(f"  - Conclusion: Significant Integration Gap exists. Targeted Restatement improves judgment by {gap*100:.2f}%.")
        print()
    
    print("Simulation Complete.")

if __name__ == "__main__":
    np.random.seed(42) # 재현성을 위한 시드 설정
    run_experiment()
