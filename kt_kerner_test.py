import kt_kernel

# Check which CPU variant was loaded
print(f"CPU variant: {kt_kernel.__cpu_variant__}")
print(f"Version: {kt_kernel.__version__}")

# Check CUDA support
from kt_kernel import kt_kernel_ext
cpu_infer = kt_kernel_ext.CPUInfer(4)
has_cuda = hasattr(cpu_infer, 'submit_with_cuda_stream')
print(f"CUDA support: {has_cuda}")

print("✓ kt-kernel installed successfully!")