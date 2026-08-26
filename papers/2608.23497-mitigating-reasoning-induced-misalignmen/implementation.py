import numpy as np

class SafetyDirectionPenalty:
    def __init__(self, num_layers, hidden_dim, lambda_sdp=0.1):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lambda_sdp = lambda_sdp
        self.d_safe = None
        self.d_reason = None
        self.key_layers = []
        
    def _generate_activations(self, n_samples, mean, std):
        """
        시뮬레이션용 활성화 벡터 생성.
        실제 모델에서는 중간층 활성화 값을 수집해야 함.
        """
        return mean + np.random.normal(0, std, (n_samples, self.hidden_dim))

    def extract_directions(self):
        """
        단계 1: 안전 방향(d_safe) 및 추론 방향(d_reason) 추출.
        유해/안전 프롬프트 및 추론 데이터에 대한 활성화의 평균 차이를 기반으로 방향을 추정합니다.
        """
        # 가상의 활성화 데이터 수집 (실제 모델 Forward Pass 결과 대체)
        # 안전 프롬프트에 대한 활성화
        act_safe = self._generate_activations(100, mean=0.0, std=1.0)
        # 유해 프롬프트에 대한 활성화 (안전성 저하를 유도하는 방향)
        act_harmful = self._generate_activations(100, mean=0.5, std=1.0)
        
        # 안전 방향: 안전 활성화의 평균 - 유해 활성화의 평균
        # (안전성이 높은 방향으로의 벡터)
        mean_safe = np.mean(act_safe, axis=0)
        mean_harmful = np.mean(act_harmful, axis=0)
        self.d_safe = mean_safe - mean_harmful
        
        # 추론 방향 추출 (추론 데이터와 비추론 데이터의 차이를 가정)
        # 여기서는 추론 능력이 향상됨에 따라 활성화가 특정 방향으로 이동한다고 가정
        act_reasoning = self._generate_activations(100, mean=0.2, std=1.0)
        act_non_reasoning = self._generate_activations(100, mean=0.0, std=1.0)
        
        mean_reasoning = np.mean(act_reasoning, axis=0)
        mean_non_reasoning = np.mean(act_non_reasoning, axis=0)
        self.d_reason = mean_reasoning - mean_non_reasoning

        # 노멀라이즈
        self.d_safe = self.d_safe / (np.linalg.norm(self.d_safe) + 1e-8)
        self.d_reason = self.d_reason / (np.linalg.norm(self.d_reason) + 1e-8)
        
        print("Direction Extraction Completed.")
        print(f"  d_safe norm: {np.linalg.norm(self.d_safe):.4f}")
        print(f"  d_reason norm: {np.linalg.norm(self.d_reason):.4f}")

    def identify_key_layers(self):
        """
        단계 1(보완): CKA를 유사한 방식으로 시뮬레이션하여 핵심 층 식별.
        실제 구현에서는 각 층의 활성화에 대해 CKA(Centered Kernel Alignment)를 계산해야 함.
        여기서는 임의로 중간 층을 핵심 층으로 선정하는 로직을 보여줍니다.
        """
        # 시뮬레이션: 각 층에 대한 '결합 강도'를 랜덤하게 생성하고 가장 강한 층을 선택
        coupling_scores = np.random.rand(self.num_layers)
        # 가장 결합이 강한 상위 20%의 층을 핵심 층으로 지정
        threshold = np.percentile(coupling_scores, 80)
        self.key_layers = [i for i, score in enumerate(coupling_scores) if score >= threshold]
        
        print(f"Identified Key Layers for SDP: {self.key_layers}")

    def compute_sdp_loss(self, activations):
        """
        단계 2: Safety-Direction Penalty (L_SDP) 계산.
        activations: dict {layer_idx: activation_vector}
        페널티는 활성화가 안전 방향(d_safe)에 얼마나 평행하게 이동했는지를 측정합니다.
        여기서는 내적(dot product)을 사용합니다.
        """
        if not self.key_layers:
            self.identify_key_layers()

        total_sdp = 0.0
        for layer_idx in self.key_layers:
            if layer_idx in activations:
                h = activations[layer_idx]
                # h와 d_safe의 내적. 값이 커질수록 안전 방향에서 더 많이 벗어나거나(부호에 따라) 이동했음을 의미함.
                # 리뷰에 따르면 "안전 방향 d_safe를 따라 얼마나 많이 이동했는지"를 페널티합니다.
                # 일반적으로 안전성이 유지되려면 특정 방향으로의 변이가 제한되어야 하므로,
                # 여기서의 내적 값의 절대값 또는 방향성에 따라 페널티를 부과합니다.
                # 리뷰 식: || h . d_safe || 가 커지는 것을 막기 위해
                proj_val = np.dot(h, self.d_safe)
                total_sdp += proj_val ** 2 # 제곱하여 비음수 페널티로 변환 (방향 무관하게 변이 억제)
        
        return (1.0 / len(self.key_layers)) * total_sdp if self.key_layers else 0.0

    def run_finetuning_simulation(self, epochs=10, learning_rate=0.01):
        """
        단계 3 및 전체 파인튜닝 루프 시뮬레이션.
        표준 Cross-Entropy Loss(시뮬레이션) + SDP Loss를 결합하여 손실을 최소화하는 과정을 보여줍니다.
        """
        print(f"\nStarting Finetuning Simulation with SDP (Lambda: {self.lambda_sdp})...")
        
        # 초기 모델 파라미터 (가상)
        model_param = np.zeros(self.hidden_dim)
        
        # 시뮬레이션용 '추론 데이터'에 대한 초기 활성화 (모델 파라미터와 연동되도록 단순화)
        # 실제 모델에서는 Forward Pass 후 hooks로 캡처
        
        history = {
            "base_loss": [],
            "sdp_loss": [],
            "total_loss": []
        }

        for epoch in range(epochs):
            # 1. Forward Pass 시뮬레이션
            # 모델 파라미터가 변함에 따라 활성화도 변한다고 가정
            # 추론 성능을 높이기 위해 모델은 d_reason 방향으로 기울기 업데이트를 시도함
            # 하지만 SDP는 d_safe 방향의 이동을 억제함
            
            # 시뮬레이션된 활성화: 모델 파라미터의 일부 + 노이즈
            # 각 핵심 층에 대한 활성화 생성
            activations = {}
            for i in range(self.num_layers):
                # 모델 파라미터와 층 인덱스 기반의 가상 활성화
                activations[i] = model_param + np.random.normal(0, 0.1, self.hidden_dim)
            
            # 2. Loss 계산
            # L_base: 추론 성능을 높이는 손실 (시뮬레이션: 모델 파라미터가 d_reason 방향으로 갈수록 감소한다고 가정)
            # 여기서는 단순히 모델 파라미터의 크기를 기준으로 한 가상의 손실을 사용하되,
            # 추론 데이터를 학습하므로 d_reason 방향으로 파라미터가 이동하는 것을 유도하는 가상의 Gradient를 적용합니다.
            
            # 가상 Base Loss Gradient: d_reason 방향으로 파라미터 업데이트
            base_grad = -self.d_reason * learning_rate
            
            # 가상 SDP Loss: compute_sdp_loss 호출
            sdp_loss_val = self.compute_sdp_loss(activations)
            
            # SDP Loss Gradient:
            # L_SDP = (1/N) * sum (h . d_safe)^2
            # dL_SDP / d_param ~ h * d_safe (단순화된 연쇄법칙 적용, h가 param에 선형 의존한다고 가정)
            # 평균을 취하므로 1/N 계수 포함
            if self.key_layers:
                # 핵심 층만 고려하여 gradient를 추정
                avg_h = np.mean([activations[i] for i in self.key_layers], axis=0)
                sdp_grad = (2.0 / len(self.key_layers)) * np.dot(avg_h, self.d_safe) * self.d_safe * learning_rate
            else:
                sdp_grad = np.zeros(self.hidden_dim)
            
            # 3. Total Loss 계산 및 기록
            # 실제 Cross-Entropy는 복잡하므로, 여기서는 '추론 정확도'를 모델하는 가상의 값을 사용
            # 모델이 d_reason 방향으로 얼마나 이동했는지 측정하여 Base Loss로 사용 (내적값이 클수록 손실 감소, 즉 성능 향상)
            # 하지만 편의상 손실값을 단순히 모델 파라미터의 d_reason 방향 투영으로 정의
            base_loss_val = -np.dot(model_param, self.d_reason) # 음수라서 성능이 좋아지면 손실은 낮아짐 (시뮬레이션용)
            
            total_loss_val = base_loss_val + self.lambda_sdp * sdp_loss_val
            
            history["base_loss"].append(base_loss_val)
            history["sdp_loss"].append(sdp_loss_val)
            history["total_loss"].append(total_loss_val)
            
            # 4. Backward Pass 및 업데이트
            # 총 Gradient = Base Grad + Lambda * SDP Grad
            total_grad = base_grad + self.lambda_sdp * sdp_grad
            
            # 파라미터 업데이트 (Gradient Descent)
            model_param = model_param - total_grad
            
            print(f"Epoch {epoch+1:2d}: Base Loss: {base_loss_val: .4f}, SDP Loss: {sdp_loss_val: .4f}, Total: {total_loss_val: .4f}")

        # 결과 분석
        final_displacement_safe = np.dot(model_param, self.d_safe)
        final_displacement_reason = np.dot(model_param, self.d_reason)
        
        print("\n--- Final Analysis ---")
        print(f"Final Displacement along d_safe: {final_displacement_safe:.4f} (Should be small/stabilized)")
        print(f"Final Displacement along d_reason: {final_displacement_reason:.4f} (Reasoning capability)")
        
        # 시뮬레이션된 결과 검증: SDP가 적용되었으므로 d_safe 방향의 이동이 억제되어야 함
        # (여기서는 랜덤 노이즈와 간단한 업데이트 규칙이므로 정량적 검증보다는 코드 흐름 확인이 목적)

if __name__ == "__main__":
    # 설정
    NUM_LAYERS = 12
    HIDDEN_DIM = 64
    LAMBDA_SDP = 0.5
    
    # 객체 생성 및 실행
    sdp_model = SafetyDirectionPenalty(NUM_LAYERS, HIDDEN_DIM, LAMBDA_SDP)
    
    # 1. 방향 벡터 추출
    sdp_model.extract_directions()
    
    # 2. 핵심 층 식별
    sdp_model.identify_key_layers()
    
    # 3. 파인튜닝 시뮬레이션 (SDP 적용)
    sdp_model.run_finetuning_simulation(epochs=10)
    
    print("\nSimulation Complete.")
