# n-hop-approach-analysis

Analysing the n-hop approach on random directed graphs.

## 기능 (Features)

| 모듈 | 설명 |
|---|---|
| `src/graph_generator.py` | 정점 수와 연결성 확률로 무방향 랜덤 그래프 생성 |
| `src/case_generator.py` | 전수 열거(`generate_strongly_connected_orientations`) 및 무작위 샘플링(`sample_strongly_connected_orientations`) 기반으로 강연결(strongly-connected) 방향 그래프 산출 |
| `src/score_calculator.py` | NumPy 기반 APSP 합계 및 n-hop 이웃 수(n=2,3,4) 계산 |
| `src/visualizer.py` | 각 점수 간 상관관계 산점도 시각화 |

## 설치 (Installation)

```bash
pip install -r requirements.txt
```

## 사용법 (Usage)

```bash
# 기본 실행 (5 정점, 연결성 60%, 랜덤 시드 없음)
python main.py

# 파라미터 지정
python main.py --vertices 5 --connectivity 0.7 --seed 42 --output result.png

# 멀티스레드/청크 사이즈 지정
python main.py --vertices 6 --connectivity 0.7 --workers 8 --chunk-size 4096 --output result.png

# 무작위 샘플링 (정점 수가 커도 일정한 시간 소요)
python main.py --vertices 10 --connectivity 0.5 --seed 42 --max-samples 500 --output result.png

# 무작위 샘플링 + 멀티스레드 병렬 실행
python main.py --vertices 10 --connectivity 0.5 --seed 42 --max-samples 500 --workers 8 --output result.png

# 멀티프로세스 병렬 실행 (GIL 우회로 CPU 바운드 처리량 향상)
python main.py --vertices 10 --connectivity 0.5 --seed 42 --max-samples 500 --workers 8 --processes --output result.png
```

### CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--vertices` | 5 | 정점 수 |
| `--connectivity` | 0.6 | 간선 존재 확률 (0–1) |
| `--seed` | None | 재현성을 위한 랜덤 시드 |
| `--output` | `result_v{N}_c{p}.png` | 저장할 이미지 경로. 미지정 시 `result_v{vertices}_c{connectivity}.png`으로 자동 생성 |
| `--workers` | CPU 코어 수 | 방향 조합 탐색용 워커 수 (전수 열거 및 샘플링 모두 사용) |
| `--chunk-size` | 2048 | 워커 작업 단위 방향 조합 수 (전수 열거 및 샘플링 모두 사용) |
| `--max-samples` | None | 무작위 샘플링 모드: 최대 N개의 강연결 방향 조합을 샘플링. 지정 시 전수 열거 대신 샘플링 사용 |
| `--processes` | False | 스레드 대신 프로세스를 사용한 병렬 실행. GIL을 우회하여 CPU 바운드 작업의 처리량을 향상 |

## 테스트 (Tests)

```bash
python -m pytest tests/ -v
```

## 프로젝트 구조 (Structure)

```
n-hop-approach-analysis/
├── main.py               # 실행 진입점
├── requirements.txt
├── src/
│   ├── graph_generator.py
│   ├── case_generator.py
│   ├── score_calculator.py
│   └── visualizer.py
└── tests/
    ├── test_graph_generator.py
    ├── test_case_generator.py
    └── test_score_calculator.py
```
