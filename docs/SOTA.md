# State of the art: fused normalization and attention kernels (late 2026)

This is a short, opinionated map of where the two kernels in this repo come
from and what production systems run today. Every reference below was opened
while writing this; numbers are quoted as the sources report them.

## 1. Performance modeling: the roofline

- **Williams, Waterman & Patterson, "Roofline: An Insightful Visual Performance
  Model for Multicore Architectures", CACM 52(4), 2009** (tech report
  UCB/EECS-2008-134).
  [EECS tech report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html).
  The model is `attainable FLOP/s = min(peak FLOP/s, peak bandwidth x operational
  intensity)`. The authors are explicit that it "need not be perfect, just
  insightful", which is how we use it here: datasheet roofs and measured roofs
  side by side, with kernels placed by their *compulsory* traffic.

## 2. Normalization kernels

- **Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019,
  [arXiv:1910.07467](https://arxiv.org/abs/1910.07467).** RMSNorm drops
  LayerNorm's re-centering (no mean subtraction, no bias). The authors report
  7%-64% lower running time than LayerNorm across models, with comparable quality. It is now the
  default norm in Llama, Qwen, Mistral, Gemma, DeepSeek.
- **Liger Kernel (LinkedIn), "Liger Kernel: Efficient Triton Kernels for LLM
  Training", [arXiv:2410.10989](https://arxiv.org/abs/2410.10989) (Oct 2024, rev. Jan
  2025).** A library of Triton kernels (RMSNorm, RoPE,
  SwiGLU, cross-entropy, fused linear cross-entropy) built on operation fusion.
  The paper reports +20% training throughput and -60% GPU memory on average
  compared with Hugging Face implementations. Its PyPI page notes an optional CuTe DSL
  backend for RMSNorm and friends ([PyPI](https://pypi.org/project/Liger-Kernel/)):
  the field is moving from Triton toward CUTLASS's Python DSL for the last few
  percent.
- **Production inference engines** (vLLM, SGLang, TensorRT-LLM) ship a
  `fused_add_rms_norm`: residual add + RMSNorm in one kernel, writing the new
  residual in place. That is the contract `rmsnorm(x, w, residual=r)`
  reproduces here (we return a new tensor instead of writing in place). Going
  further, engines fuse the norm with the *following* op (FP8/FP4 quantization of
  the output, or the QKV GEMM prologue) so the normalized activations never touch
  DRAM. This repo does not do that.

## 3. Attention kernels

- **Milakov & Gimelshein (NVIDIA), "Online normalizer calculation for softmax",
  [arXiv:1805.02867](https://arxiv.org/abs/1805.02867), 2018.** A one-pass softmax:
  keep a running max and a running sum and rescale the sum when the max changes.
  The paper reports 4 -> 3 memory accesses per element for softmax, up to 1.3x for
  softmax and 5x for softmax+TopK. This identity is the mathematical core of every
  FlashAttention version.
- **Dao, Fu, Ermon, Rudra & Re, "FlashAttention: Fast and Memory-Efficient Exact
  Attention with IO-Awareness", NeurIPS 2022,
  [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).** Tiles Q/K/V through
  on-chip SRAM so the N x N matrix is never written to HBM. The paper proves
  `O(N^2 d^2 / M)` HBM accesses (M = SRAM size), versus `Omega(Nd + N^2)` for
  standard attention.
- **Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work
  Partitioning", ICLR 2024, [arXiv:2307.08691](https://arxiv.org/abs/2307.08691).**
  It makes three changes: (1) fewer non-matmul FLOPs (deferred rescaling/normalization),
  (2) parallelism over the sequence dimension as well as batch x heads,
  (3) Q split across warps while K/V are shared, which removes the inter-warp
  reduction. It reports ~2x over FA1 and up to 73% of A100 peak in the forward pass
  ([blog](https://crfm.stanford.edu/2023/07/17/flash2.html)). **The kernel in this
  repo is an FA2-style forward pass.**
- **Shah, Bikshandi, Zhang, Thakkar, Ramani & Dao, "FlashAttention-3: Fast and
  Accurate Attention with Asynchrony and Low-precision",
  [arXiv:2407.08608](https://arxiv.org/abs/2407.08608), 2024.** Hopper-specific:
  warp specialization, TMA, async WGMMA, and softmax overlapped with the GEMMs
  ("ping-pong"). It also adds FP8 with incoherent processing. Reported numbers: FA2 reaches only 35% of
  H100 peak; FA3 gets 1.5-2.0x more, up to 740 TFLOP/s FP16 (75%) and close
  to 1.2 PFLOP/s FP8.
- **FlashAttention-4, "Algorithm and Kernel Pipelining Co-Design for Asymmetric
  Hardware Scaling", [arXiv:2603.05451](https://arxiv.org/abs/2603.05451), 2026**
  ([author blog](https://tridao.me/blog/2026/flash4/)). This one targets datacenter
  Blackwell (B200/GB200), where tensor-core throughput doubled over Hopper but the
  exponential units and shared-memory bandwidth did not. Its answers are
  software-emulated `exp2` on FMA units, conditional softmax rescaling, and
  2-CTA MMA with tensor memory. It is reported at ~1600 TFLOP/s (~71%) on B200.
  Secondary sources say FA4 relies on `tcgen05`/TMEM instructions that consumer
  Blackwell (sm_120, our GPU) does not have, so it does not run on GeForce parts.
  We have not verified that ourselves.
- **Ye et al., "FlashInfer: Efficient and Customizable Attention Engine for LLM
  Inference Serving", MLSys 2025, [arXiv:2501.01005](https://arxiv.org/abs/2501.01005).**
  The serving view of attention: block-sparse / paged KV-cache formats, JIT-compiled
  attention variants from a template, and a load-balanced scheduler compatible with
  CUDA Graphs. It is integrated into SGLang, vLLM and MLC-Engine. The prefill kernel is
  FA2/FA3-like; decode (1 query token) is memory-bound and uses split-KV instead.
- **Triton's official tutorial "Fused Attention"**
  ([triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html))
  implements FA2 in Triton. Our kernel follows the same two-stage causal
  structure (an unmasked off-diagonal loop plus a masked diagonal block) but is
  written from scratch and stripped to the forward pass.

## 4. What this means on the GPU used here

- **GPU:** GeForce RTX 5070 Ti **Laptop** (GB205, sm_120): 46 SMs, 5888 CUDA cores,
  184 tensor cores, 36 MB L2 (driver-reported `L2_cache_size`; the spec sheet below says 40 MB), 12 GB GDDR7 on a 192-bit bus at 28 Gbps = **672 GB/s**,
  ~1.82 GHz boost (80 W variant), FP16 tensor **~55 TFLOP/s dense with FP32
  accumulate** (110 with FP16 accumulate), 21.4 TFLOP/s FP32
  ([hmc-tech spec sheet](https://hmc-tech.com/gpus/nvidia-geforce-rtx-5070-ti-laptop-80w),
  [Notebookcheck](https://www.notebookcheck.net/Nvidia-GeForce-RTX-5070-Ti-Laptop-Benchmarks-and-Specs.934945.0.html)).
  Note: one NVIDIA comparison page lists 896 GB/s for the laptop part. That figure
  belongs to the 256-bit *desktop* RTX 5070 Ti and conflicts with the 192-bit bus every
  other source (and `nvidia-smi`) reports. Laptop clocks also depend on the OEM power
  limit, so we **measure** both roofs instead of trusting the datasheet.
- **sm_120 has no WGMMA/TMA-multicast/tcgen05 path** of the kind FA3/FA4 rely on.
  Triton lowers `tl.dot` to warp-level `mma.sync` here, which is FA2's programming
  model. So FA2 is the right algorithm to study on this hardware, not a legacy
  choice.
- **Nsight Compute:** official Blackwell support starts with Nsight Compute
  2025.1 (shipped with CUDA 12.8,
  [NVIDIA blog](https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/);
  [release notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html)).
  The machine used here only has 2024.1.1; see the README for what happened when we
  tried it.
