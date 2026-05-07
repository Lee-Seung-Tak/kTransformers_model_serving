**시작하기에 앞서** KTransformers가 뭐냐?

    거대 MoE 모델을 GPU VRAM이 부족한 환경에서 굴리기 위한 CPU/GPU 이종 추론 프레임워크다.

   핵심 아이디어는 Attention/dense 레이어와 자주 쓰이는 expert만 GPU에 올리고, 나머지 expert들은 CPU의 DRAM에 두고 
   CPU에서 직접 연산해서 합치는 방식
   ---

# 파이썬 버전은 
Python 3.10, 3.11 또는 3.12를 요구하나, 3.12로 맞춤.

# 가상 환경 활성화
```bash
1. python3 -m venv kt_kernel_env 

2. source kt_kernel_env/bin/activate
```
---
# model down
```
hf download Qwen/Qwen3.5-122B-A10B-FP8 --local-dir Qwen3.5-122B-A10B-FP8
```

# kt_kerner docs
```
https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md
```
---

# kt_kerner 설치
```bash
pip install kt_kernel
```
---
# kt_kerner 설치 테스트
```bash
python kt_kerner_test.py
```
---
# kt_kerner 설치 테스트 에러 발생 시 
```code
(kt_kernel_env) lst@SeungTak-Lee:~/kTransformers_model_serving$ python kt_kerner_test.py 
Traceback (most recent call last):
  File "/home/lst/kTransformers_model_serving/kt_kernel_env/lib/python3.10/site-packages/kt_kernel/_cpu_detect.py", line 228, in load_extension
    ext = importlib.util.module_from_spec(spec)
  File "<frozen importlib._bootstrap>", line 571, in module_from_spec
  File "<frozen importlib._bootstrap_external>", line 1176, in create_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
ImportError: libhwloc.so.15: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/lst/kTransformers_model_serving/kt_kerner_test.py", line 1, in <module>
    import kt_kernel
  File "/home/lst/kTransformers_model_serving/kt_kernel_env/lib/python3.10/site-packages/kt_kernel/__init__.py", line 41, in <module>
    _kt_kernel_ext, __cpu_variant__ = _initialize_cpu()
  File "/home/lst/kTransformers_model_serving/kt_kernel_env/lib/python3.10/site-packages/kt_kernel/_cpu_detect.py", line 289, in initialize
    ext = load_extension(variant)
  File "/home/lst/kTransformers_model_serving/kt_kernel_env/lib/python3.10/site-packages/kt_kernel/_cpu_detect.py", line 259, in load_extension
    raise ImportError(
ImportError: Failed to load kt_kernel extension (variant: avx2). Original error: libhwloc.so.15: cannot open shared object file: No such file or directory
This usually means the kt_kernel package is not properly installed.
```

**ImportError: libhwloc.so.15: cannot open shared object file: No such file or directory** 여기만 보면 됨.
```
https://stackoverflow.com/questions/77562062/libhwloc-so-5-error-while-installing-vortex 참고
```
install libhwloc-dev 설치 

```bash
sudo apt-get update
sudo apt-get install libhwloc-dev -y
```
---

**다른 에러 발생하는 경우**
```bash
(kt_kernel_env) lst@SeungTak-Lee:~/kTransformers_model_serving$ python kt_kerner_test.py 
CPU variant: avx2
Version: 0.5.0
Illegal instruction (core dumped)
```
찾아보니, 성능 극대화를 위해서 CPU의 특수 명령어를 사용하도록 컴파일 되는데 본인 컴퓨터가 i7 12700kf여서 P코어랑 E코어 섞인 하이브리드 구조면 안됨.
DOCS처럼
```bash
# Override automatic CPU detection (for testing or debugging)
export KT_KERNEL_CPU_VARIANT=avx2  # Force specific variant

# Enable debug output to see detection process
export KT_KERNEL_DEBUG=1
python -c "import kt_kernel"
```
하면 됨. 그래도 안된다? 맘 편하게 그냥 빌드하자.
```bash
# 1. 기존 패키지 삭제
pip uninstall kt-kernel -y

# 2. 소스 다운로드 및 빌드 설치
git clone https://github.com/kvcache-ai/ktransformers.git
pip install --upgrade pip setuptools wheel ninja cmake 

cd ktransformers
git submodule update --init --recursive # -> 해당 프로젝트가 의존하고 있는 다른 외부 프로젝트까진 전부다 가져와버려

# 3. 현재 CPU 환경에 맞춰 빌드 설치 -> 시간 좀 걸림
pip install .
```

와 이젠 빌드에서 에러가 나버린다?
```bash
ERROR: Could not find a version that satisfies the requirement kt-kernel==0.6.1 (from ktransformers) (from versions: 0.4.4, 0.5.0)
ERROR: No matching distribution found for kt-kernel==0.6.1
```

빌드 도구를 설치해보자.
```bash
sudo apt update
sudo apt install build-essential cmake ninja-build g++ -y
```

kt-kernel 로컬을 설치해보자
```bash
pip install ./kt-kernel
```

커널 설치 문제 없으니까 메인 패키지 설치해보자
```bash
pip install . --no-deps
```
그리고 다시
```bash
export KT_KERNEL_CPU_VARIANT=avx2
export KT_KERNEL_DEBUG=1

python kt_kerner_test.py
```

그럼 짜잔
```bash

((kt_kernel_env) ) lst@SeungTak-Lee:~/kTransformers_model_serving$ python kt_kerner_test.py
[kt-kernel] Using environment override: avx2
[kt-kernel] Selected CPU variant: avx2
[kt-kernel] Multi-variant avx2 not found, using single-variant build
[kt-kernel] Loading avx2 from: /home/lst/kTransformers_model_serving/kt_kernel_env/lib/python3.12/site-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so
[kt-kernel] Successfully loaded AVX2 variant
[kt-kernel] Extension module loaded: kt_kernel_ext
CPU variant: avx2
Version: 0.6.1
CPUInfer[0x28552090]: Hello
WorkerPool[0x28e208c0] 1 subpools, [numa:threads][0:4] 
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 4 threads
CUDA support: True
✓ kt-kernel installed successfully!
CPUInfer[0x28552090]: Goodbye
((kt_kernel_env) ) lst@SeungTak-Lee:~/kTransformers_model_serving$ 

```

# kt version 확인
```bash
kt version

[kt-kernel] Using environment override: avx2
[kt-kernel] Selected CPU variant: avx2
[kt-kernel] Multi-variant avx2 not found, using single-variant build
[kt-kernel] Loading avx2 from: /home/solideos/kTransformers_model_serving/kt_kernel_env/lib/python3.12/site-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so
[kt-kernel] Successfully loaded AVX2 variant
[kt-kernel] Extension module loaded: kt_kernel_ext

KTransformers CLI v0.6.1.post1

  Python      3.12.3
  Platform    Linux 6.6.87.2-microsoft-standard-WSL2
  CUDA        13.1

Packages:

  kt-kernel    0.6.1.post1
  sglang-kt    Not installed


SGLang is not installed

Install SGLang (kvcache-ai fork) using one of these methods:

Option A - One-click install (recommended):
   From the ktransformers root directory, run:
   ./install.sh

Option B - pip install:
   pip install sglang-kt

Option C - From source:
   git clone --recursive https://github.com/kvcache-ai/ktransformers.git
   cd ktransformers
   pip install "third_party/sglang/python"

Note: Make sure to run these commands in the correct Python environment

```

자 그럼 추천 옵션인 ./install.sh로 sglang-kt를 설치하자.

```bash
 kt version

KTransformers CLI v0.6.1.post1

  Python      3.12.3
  Platform    Linux 6.6.87.2-microsoft-standard-WSL2
  CUDA        13.1

Packages:

  kt-kernel    0.6.1.post1
  sglang-kt    0.6.1.post1 (sglang-kt)
```

확인됨

# 만약 내 컴퓨터가 h100이다 , 여러대이다, 그런데 빌드 시 nvcc이슈로 안된다고 한다?
```
nvcc fatal : Unsupported gpu architecture 'compute_89'  -> nvcc 버전이 너무 낮아서 그렇다.

cat >> ~/.bashrc << 'EOF'

# CUDA 12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
EOF

source ~/.bashrc

이렇게 해서 12.8로 고정해주자. 그럼 됨.
```

그리고
```bash
cd ~/kTransformers_model_serving/ktransformers

# 이전 빌드 흔적 제거 (중요! 안 지우면 옛날 결과 캐시됨)
rm -rf kt-kernel/build kt-kernel/*.egg-info

# CMake에 새 nvcc 경로 명시
export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc"

# 재빌드
pip install ./kt-kernel --no-build-isolation
```
요로코롬 빌드 캐쉬 지우고 다시 해보자 그럼 된다.
---

# 모델 서빙해보자
``` bash
# ============================================
# Qwen3.5-122B-A10B 서빙 (LLAMAFILE 백엔드)
# ============================================

python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /data/ai/models/Qwen3.5-122B-A10B \
  --kt-method LLAMAFILE \
  --kt-weight-path /data/ai/models/Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL \
  --kt-cpuinfer 24 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 128 \
  --kt-max-deferred-experts-per-token 2 \
  --trust-remote-code \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 \
  --served-model-name Qwen3.5-122B-A10B \
  --enable-mixed-chunk \
  --tensor-parallel-size 2 \
  --enable-p2p-check \
  --disable-shared-experts-fusion \
  --reasoning-parser qwen3

[ 해석 ]
python -m sglang.launch_server \
  --host 0.0.0.0 \                                                              # 외부에서 접속 가능하게 모든 IP 허용
  --port 8000 \                                                                 # API 포트 (필요시 변경)
  --model /data/ai/models/Qwen3.5-122B-A10B \                                   # GPU용 원본 모델 (config + safetensors)
  --kt-method LLAMAFILE \                                                       # CPU 백엔드 종류 (AMX 없을때 쓰는거, GGUF 사용)
  --kt-weight-path /data/ai/models/Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL \          # CPU expert가 읽을 GGUF 가중치 경로
  --kt-cpuinfer 24 \                                                            # CPU 추론 스레드 = 물리 코어 수 (48코어/2HT = 24)
  --kt-threadpool-count 2 \                                                     # 스레드풀 개수 = NUMA 노드 수 (dual socket)
  --kt-num-gpu-experts 128 \                                                    # GPU에 올릴 expert 수 (총 256개 중 절반)
  --kt-max-deferred-experts-per-token 2 \                                       # 파이프라인 실행용 지연 expert 수 (CPU/GPU 동시 처리)
  --trust-remote-code \                                                         # Qwen 커스텀 모델 코드 신뢰 (필수)
  --mem-fraction-static 0.90 \                                                  # GPU 메모리의 90% 사용 (KV cache 등)
  --chunked-prefill-size 4096 \                                                 # 긴 입력을 4096 토큰씩 쪼개서 처리
  --served-model-name Qwen3.5-122B-A10B \                                       # API 호출시 model 필드에 쓸 이름
  --enable-mixed-chunk \                                                        # prefill + decode 섞어서 처리 (throughput 향상)
  --tensor-parallel-size 2 \                                                    # H100 2장에 텐서 분산
  --enable-p2p-check \                                                          # GPU간 P2P 통신 가능 여부 체크
  --disable-shared-experts-fusion                                               # shared expert 융합 비활성화 (KT와 호환성 위해)

  ---
  LLAMAFILE말고  FP8로 돌리려면
  LLAMAFILE = lamacpp기반의 범용 CPU 백엔드 사용해서 느림
  FP8:       AVX512_BF16 (한 번에 32개 곱셈)
  python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /data/ai/models/Qwen3.5-122B-A10B-FP8 \
  --kt-method FP8 \
  --kt-weight-path /data/ai/models/Qwen3.5-122B-A10B-FP8 \
  --kt-cpuinfer 24 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 24 \
  --kt-max-deferred-experts-per-token 2 \
  --kt-gpu-prefill-token-threshold 2048 \
  --kt-enable-dynamic-expert-update \
  --kt-expert-placement-strategy uniform \
  --trust-remote-code \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 \
  --served-model-name Qwen3.5-122B-A10B \
  --enable-mixed-chunk \
  --tensor-parallel-size 2 \
  --enable-p2p-check \
  --disable-shared-experts-fusion \
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
```
위가 된다면
```
# ============================================
# 1단계: BF16 -> AMXINT8 가중치 변환 (한 번만)
# ============================================

python ~/kTransformers_model_serving/ktransformers/kt-kernel/scripts/convert_cpu_weights.py \
  --input-path /data/ai/models/Qwen3.5-122B-A10B \
  --input-type bf16 \
  --output /data/ai/models/Qwen3.5-122B-A10B-INT8 \
  --quant-method int8


# ============================================
# 2단계: AMXINT8 백엔드로 서빙
# ============================================

python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /data/ai/models/Qwen3.5-122B-A10B \
  --kt-method AMXINT8 \
  --kt-weight-path /data/ai/models/Qwen3.5-122B-A10B-INT8 \
  --kt-cpuinfer 24 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 32 \
  --kt-max-deferred-experts-per-token 2 \
  --trust-remote-code \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 \
  --served-model-name Qwen3.5-122B-A10B \
  --enable-mixed-chunk \
  --tensor-parallel-size 2 \
  --enable-p2p-check \
  --disable-shared-experts-fusion


[ 해석 ]

# ============================================
# 1단계: BF16 -> AMXINT8 가중치 변환 (한 번만)
# ============================================
# tmux 안에서 돌리는걸 추천 (30분~1시간 걸림)

python ~/kTransformers_model_serving/ktransformers/kt-kernel/scripts/convert_cpu_weights.py \
  --input-path /data/ai/models/Qwen3.5-122B-A10B \                              # 원본 BF16 모델 위치
  --input-type bf16 \                                                           # 원본 dtype
  --output /data/ai/models/Qwen3.5-122B-A10B-INT8 \                             # 변환된 INT8 저장 경로 (~122GB)
  --quant-method int8                                                           # 양자화 방식 (int4도 가능하지만 정확도 ↓)


# ============================================
# 2단계: AMXINT8 백엔드로 서빙 (변환 끝난 후)
# ============================================

python -m sglang.launch_server \
  --host 0.0.0.0 \                                                              # 외부 접속 허용
  --port 8000 \                                                                 # API 포트
  --model /data/ai/models/Qwen3.5-122B-A10B \                                   # GPU용 원본 (config 읽고 GPU expert 가중치 가져감)
  --kt-method AMXINT8 \                                                         # AMX INT8 백엔드 (Sapphire Rapids 가속)
  --kt-weight-path /data/ai/models/Qwen3.5-122B-A10B-INT8 \                     # 변환된 INT8 가중치 (LLAMAFILE때랑 다름!)
  --kt-cpuinfer 24 \                                                            # 물리 코어 수
  --kt-threadpool-count 2 \                                                     # NUMA 노드 수
  --kt-num-gpu-experts 128 \                                                    # GPU에 올릴 expert 수
  --kt-max-deferred-experts-per-token 2 \                                       # 파이프라인 지연 expert
  --trust-remote-code \                                                         # 커스텀 모델 코드 허용
  --mem-fraction-static 0.90 \                                                  # GPU 메모리 90% 사용
  --chunked-prefill-size 4096 \                                                 # prefill chunk size
  --served-model-name Qwen3.5-122B-A10B \                                       # API에서 부를 모델명
  --enable-mixed-chunk \                                                        # prefill+decode 혼합
  --tensor-parallel-size 2 \                                                    # H100 2장 분산
  --enable-p2p-check \                                                          # P2P 통신 체크
  --disable-shared-experts-fusion                                               # shared expert 융합 OFF
```
이렇게 해보자

*동작테스트*는 이렇게
```
# 모델 리스트 확인 (서버 정상 동작 체크)
curl http://localhost:8000/v1/models

# 간단한 추론 테스트
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5",
    "messages": [{"role": "user", "content": "안녕, 한국어로 짧게 답해줘"}],
    "max_tokens": 1000
  }' | python3 -m json.tool
```

종료는 pkill -f "sglang.launch_server"
