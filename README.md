# Paper Reviews & Implementations

AI/ML 논문 리뷰와 핵심 아이디어의 간이 구현을 모아둔 저장소입니다.

- 리뷰 전문은 [블로그](https://github.com/arthur-yoon/paper-reviews)에서 각 논문 폴더의 `README.md`를 확인하세요.
- 구현은 논문의 핵심 아이디어를 이해하기 위한 **최소 예제**이며, 프로덕션 품질을 목표로 하지 않습니다.
- 자동 생성 파이프라인: [content-engine](https://github.com/arthur-yoon) (Qwen3.8-27B, 로컬 실행)

## 구조

```
papers/
  <arxiv-id>-<slug>/
    README.md      # 리뷰 (문제/방법/실험/한계/적용 가능성)
    implementation.py  # 간이 구현
```
