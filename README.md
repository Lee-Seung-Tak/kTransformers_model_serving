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
