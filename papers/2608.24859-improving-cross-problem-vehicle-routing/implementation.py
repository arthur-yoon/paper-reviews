import numpy as np

# ==============================================================================
# 1. VRP 변형 문제 정의 및 로컬 서치 (POLAR의 핵심)
# ==============================================================================

class VRPVariant:
    """
    VRP 변형 문제를 정의합니다.
    각 변형은 고유한 제약 조건(예: 시간 창, 용량)을 가질 수 있습니다.
    """
    def __init__(self, name, num_vehicles, num_customers, seed=0):
        self.name = name
        self.num_vehicles = num_vehicles
        self.num_customers = num_customers
        np.random.seed(seed)
        
        # 0: Depot, 1..N: Customers
        self.depot = np.array([0.0, 0.0])
        self.customers = np.random.rand(num_customers, 2) * 10
        self.all_nodes = np.vstack([self.depot, self.customers])
        
        # 변형에 따른 제약 파라미터 (예: 최대 거리, 시간 등)
        self.max_capacity = np.random.randint(10, 50)
        self.time_window = np.random.rand(num_customers, 2) * 10  # [start, end]

    def compute_distance_matrix(self):
        """Euclidean distance matrix 계산"""
        nodes = self.all_nodes
        diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

    def evaluate_solution(self, route_indices):
        """
        주어진 경로 인덱스 배열의 비용(거리)을 계산합니다.
        route_indices: [0, 3, 1, 0] 형태의 인덱스 배열 (Depot으로 시작/종료)
        """
        dist_matrix = self.compute_distance_matrix()
        cost = 0.0
        for i in range(len(route_indices) - 1):
            cost += dist_matrix[route_indices[i], route_indices[i+1]]
        return cost

    def local_search_or_opt(self, route_indices, max_iterations=10):
        """
        POLAR의 핵심: 로컬 서치 (Or-Opt 2) 후처리.
        경로 내 2개 노드를 선택하여 순서를 바꾸거나 이동시켜 비용을 줄이려는 시도.
        """
        current_route = list(route_indices)
        current_cost = self.evaluate_solution(current_route)
        dist_matrix = self.compute_distance_matrix()
        
        for _ in range(max_iterations):
            improved = False
            n = len(current_route)
            # Or-Opt 2: 두 인접한 노드를 다른 위치로 이동
            for i in range(1, n - 2):
                for j in range(1, n - 1):
                    if j == i or j == i - 1: continue
                    
                    # 노드 i, i+1 제거
                    moved_nodes = [current_route[i], current_route[i+1]]
                    new_route_temp = current_route[:i] + current_route[i+2:]
                    
                    # j 위치에 삽입
                    if j > i: j -= 2 # 인덱스 조정
                    
                    candidate_route = new_route_temp[:j] + moved_nodes + new_route_temp[j:]
                    
                    # depot(0)은 항상 첫/마지막 위치 유지 보장
                    if candidate_route[0] != 0 or candidate_route[-1] != 0:
                        continue
                        
                    cand_cost = self.evaluate_solution(candidate_route)
                    if cand_cost < current_cost - 1e-6:
                        current_route = candidate_route
                        current_cost = cand_cost
                        improved = True
                        break
                if improved: break
            
            if not improved:
                break
                
        return current_route, current_cost


# ==============================================================================
# 2. PLE 인코더 아키텍처 (Progressive Layered Extraction)
# ==============================================================================

class ExpertLayer:
    """
    단일 전문가 층: 입력을 변환하는 선형층 + 활성화
    """
    def __init__(self, input_dim, hidden_dim):
        self.W = np.random.randn(input_dim, hidden_dim) * (1.0 / np.sqrt(input_dim))
        self.b = np.zeros(hidden_dim)

    def forward(self, x):
        return np.tanh(x @ self.W + self.b)


class PLEEncoderLayer:
    """
    PLE 인코더의 한 층.
    - 공유 전문가 (Shared Expert)
    - 태스크별 전문가 (Task-specific Experts)
    - 게이팅 메커니즘 (Gating Mechanism)
    """
    def __init__(self, input_dim, output_dim, num_tasks):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tasks = num_tasks
        
        # 전문가들
        self.shared_expert = ExpertLayer(input_dim, output_dim)
        self.task_experts = [ExpertLayer(input_dim, output_dim) for _ in range(num_tasks)]
        
        # 게이팅: 각 입력 노드에 대해 전문가 가중치 생성
        # 가중치 벡터 크기: num_experts = 1 (shared) + num_tasks
        self.gate_W = np.random.randn(input_dim, 1 + num_tasks) * (1.0 / np.sqrt(input_dim))
        self.gate_b = np.zeros(1 + num_tasks)

    def forward(self, x, task_id):
        """
        x: (batch_size, seq_len, input_dim)
        task_id: 해당 배치의 태스크 ID (배치 차원)
        """
        B, L, D = x.shape
        
        # 1. 전문가 출력 계산
        shared_out = self.shared_expert.forward(x) # (B, L, O)
        task_outs = [expert.forward(x) for expert in self.task_experts] # List of (B, L, O)
        
        # 2. 게이팅 가중치 계산
        # gate_logits: (B, L, num_experts)
        gate_logits = x @ self.gate_W + self.gate_b
        # Softmax을 전문가 축(axis=-1)으로 수행
        gate_weights = np.exp(gate_logits - np.max(gate_logits, axis=-1, keepdims=True))
        gate_weights /= np.sum(gate_weights, axis=-1, keepdims=True)
        
        # 3. 가중 합 (Weighted Sum)
        # shared_out과 task_outs를 합쳐서 (B, L, O, num_experts) 형태로 만들고 가중치 적용
        # shared expert는 index 0, task experts는 index 1..N
        
        # 효율성을 위해 직접 계산
        output = np.zeros((B, L, self.output_dim))
        
        # Shared expert 가중치 (B, L, 1)
        w_shared = gate_weights[:, :, 0:1]
        output += w_shared * shared_out
        
        # Task experts 가중치
        # 현재 배치의 task_id에 해당하는 전문가만 사용한다고 가정하되,
        # 실제 PLE에서는 게이트가 모든 전문가에 대한 가중치를 결정합니다.
        # 여기서 간소화를 위해, 해당 태스크의 전문가 가중치만 추출하거나 
        # 모든 전문가 가중치를 사용하도록 구현합니다.
        # 리뷰에서 "게이팅 메커니즘이 어떤 정보를 공유, 어떤 것을 태스크별로 보낼지 판단"한다고 했으므로
        # Softmax으로 나온 가중치를 그대로 사용함.
        
        for i, task_out in enumerate(task_outs):
            w_task = gate_weights[:, :, i+1:i+2]
            output += w_task * task_out
            
        return output


class PLEMultiTaskEncoder:
    """
    다중 층 PLE 인코더
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_tasks):
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.layers = [PLEEncoderLayer(input_dim if i == 0 else hidden_dim, hidden_dim, num_tasks) for i in range(num_layers)]

    def forward(self, x, task_ids):
        """
        x: (Batch, SeqLen, InputDim)
        task_ids: (Batch,) 배열
        """
        h = x
        for layer in self.layers:
            h = layer.forward(h, task_ids)
        return h


# ==============================================================================
# 3. POLAR 알고리즘 구현 (선호 최적화 시뮬레이션)
# ==============================================================================

class POLARTrainer:
    """
    POLAR: Preference Optimization with Locally Augmented Refinement
    실제 경사 하강법 대신, 로컬 서치로 만든 선호 쌍을 기준으로 
    모델 파라미터를 업데이트하는 로직을 시뮬레이션합니다.
    """
    def __init__(self, encoder, vrp_variants):
        self.encoder = encoder
        self.variants = vrp_variants
        self.num_tasks = len(vrp_variants)
        
    def generate_policy_solution(self, variant, task_id, batch_size=1, seq_len=5):
        """
        정책(이 예제에서는 임의의 초기 경로 + 인코더의 영향력 시뮬레이션)이 생성하는 해.
        실제 모델에서는 이 부분이 순전파(Forward Pass)입니다.
        여기서는 인코더 출력을 활용해 '의사 경로'를 생성하거나, 
        단순히 변형의 특성(예: customer 수)을 반영하는 임의 경로 생성.
        """
        np.random.seed(task_id * 100 + batch_size)
        # Depot(0) 포함하여 고객들을 무작위로 선택 (순수 VRP 시뮬레이션을 위한 간단화)
        num_customers_to_visit = min(5, variant.num_customers)
        customers = np.random.choice(range(1, variant.num_customers + 1), num_customers_to_visit, replace=False)
        route = [0] + list(customers) + [0]
        return route

    def train_step(self, epoch=5, batch_size=4):
        """
        POLAR 학습 스텝 시뮬레이션
        """
        print(f"Starting POLAR Training with PLE Encoder...")
        print(f"Tasks (VRP Variants): {len(self.variants)}")
        
        for e in range(epoch):
            total_gap_improvement = 0.0
            for t_id, variant in enumerate(self.variants):
                # 1. 정책으로 해 생성 (원래 해)
                orig_route = self.generate_policy_solution(variant, t_id, batch_size)
                orig_cost = variant.evaluate_solution(orig_route)
                
                # 2. POLAR 핵심: 로컬 서치 적용 (개선된 해)
                refined_route, refined_cost = variant.local_search_or_opt(orig_route, max_iterations=5)
                
                # 3. 선호 쌍 (Preference Pair) 형성
                # 원본 해 (Bad) vs 로컬 서치 해 (Good)
                margin = orig_cost - refined_cost
                total_gap_improvement += margin
                
                # 4. PLE 인코더 포워드 패스 시뮬레이션
                # 인코더는 표현을 학습합니다. 여기서는 가중치 업데이트 시뮬레이션
                input_dim = 4 # 예: [x, y, capacity, time]
                seq_len = variant.num_customers
                dummy_input = np.random.randn(batch_size, seq_len, input_dim)
                
                # 인코더가 표현을 분리하는 과정
                encoded_repr = self.encoder.forward(dummy_input, np.full(batch_size, t_id, dtype=int))
                
                # 5. (시뮬레이션) 인코더 가중치 업데이트
                # 실제론 encoded_repr과 refined_cost의 차이를 기반으로 그래디언트 계산
                # 여기서는 가중치에 작은 노이즈를 추가하여 학습 진행 시뮬레이션
                for layer in self.encoder.layers:
                    layer.gate_W += 0.01 * np.random.randn(*layer.gate_W.shape)
                    # expert weights update simulation
                    layer.shared_expert.W += 0.01 * np.random.randn(*layer.shared_expert.W.shape)
                    for exp in layer.task_experts:
                        exp.W += 0.01 * np.random.randn(*exp.W.shape)

            avg_margin = total_gap_improvement / len(self.variants)
            print(f"Epoch {e+1}/{epoch} - Avg Preference Margin (Gap Improved): {avg_margin:.4f}")

# ==============================================================================
# 4. 메인 실행 및 테스트
# ==============================================================================

def main():
    # 1. 다양한 VRP 변형 생성 (In-distribution)
    variants = [
        VRPVariant("Variant_A_Capacity", num_vehicles=3, num_customers=10, seed=1),
        VRPVariant("Variant_B_TimeWindow", num_vehicles=5, num_customers=15, seed=2),
        VRPVariant("Variant_C_Mixed", num_vehicles=2, num_customers=8, seed=3),
        VRPVariant("Variant_D_Unseen", num_vehicles=4, num_customers=12, seed=4) # OOD 시뮬레이션
    ]
    
    # 2. PLE 인코더 초기화
    # Input Dim: 노드 특성 (x, y, capacity, time_start, time_end)
    input_dim = 5 
    hidden_dim = 32
    num_layers = 3
    num_tasks = len(variants)
    
    encoder = PLEMultiTaskEncoder(input_dim, hidden_dim, num_layers, num_tasks)
    
    # 3. POLAR 트레이너 초기화 및 학습
    trainer = POLARTrainer(encoder, variants)
    
    print("-" * 30)
    # 학습 실행
    trainer.train_step(epoch=10)
    print("-" * 30)
    
    # 4. 추론 및 일반화 테스트 시뮬레이션 (Out-of-Distribution)
    print("\nTesting Generalization on Unseen Variants...")
    unseen_variant = VRPVariant("Unseen_Variant_E", num_vehicles=6, num_customers=20, seed=99)
    
    # 인코더가 새로운 태스크(예: task_id = 0을 재사용하거나, 
    # 실제론 새로운 expert가 필요하지만, 여기서는 기존 expert의 가중치로 처리)
    # PLE의 핵심은 표현 분리가므로, 같은 구조의 input으로 forward pass 수행
    dummy_input_ood = np.random.randn(1, 20, input_dim)
    encoded_ood = encoder.forward(dummy_input_ood, np.array([0])) # Task 0의 expert를 활용해 유사성 테스트
    
    # 로컬 서치로 해결 품질 측정
    route = trainer.generate_policy_solution(unseen_variant, 0, 1)
    orig_cost = unseen_variant.evaluate_solution(route)
    refined_route, refined_cost = unseen_variant.local_search_or_opt(route, max_iterations=10)
    
    print(f"Unseen Variant Base Cost: {orig_cost:.4f}")
    print(f"Unseen Variant Refined Cost (after Local Search): {refined_cost:.4f}")
    print(f"Gap Reduction: {((orig_cost - refined_cost) / orig_cost) * 100:.2f}%")
    
    print("\nImplementation Complete. (POLAR + PLE Encoder)")

if __name__ == "__main__":
    main()
