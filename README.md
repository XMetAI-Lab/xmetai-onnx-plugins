# XMetAI ONNX Plugins

`xmetai-onnx-plugins` provides custom ONNX Runtime operators used by XMetAI
deployment and export workflows.

The package currently builds the native extension module
`xmetai_onnx_plugins`, including:

- `SelectiveScanFn`: CUDA implementation of Mamba-style selective scan for
  ONNX Runtime.
- `Mamba3SisoFn`: CUDA implementation of the Mamba-3 SISO forward path for
  ONNX Runtime. This v1 path supports forward inference only and does not
  support recurrent input/final states or packed variable-length `cu_seqlens`.
- `GatedDeltaNet2Fn`: CUDA implementation of the Gated DeltaNet-2 recurrent
  forward path used by `fla.ops.gdn2.chunk_gdn2` during ONNX Runtime
  inference. This v1 path supports dense `(batch, seqlen, head, dim)` tensors
  and optional initial state, but does not return final states or support
  packed variable-length `cu_seqlens`.
- `SoftmaxCPU`: CPU-side softmax custom operator support.

### GDN-2 execution

For key dimensions up to 256, each CUDA warp processes an independent value
column and holds its recurrent state in fixed-size register arrays. This path
does not allocate a global state workspace or use block-wide barriers. Larger
key dimensions retain the generic implementation, with per-invocation state
scratch allocated through ONNX Runtime's CUDA allocator and returned to it by
RAII. No mutable state buffer is cached on the kernel object; concurrent calls
and dynamic batch sizes get independent allocations. Both paths use
FP32 and preserve the existing operator schema, optional initial state, scale,
and Q/K normalization attributes; existing ONNX graphs need no re-export.

This is a fused recurrent implementation, not the FLA chunk algorithm: tokens
remain sequential within each warp. Warp reductions change floating-point
summation order. GPU numerical validation and performance measurements are
required before deploying a rebuilt plugin; no speedup is guaranteed by the
source change alone. The generic path uses ORT's stream-aware scratch API
(available since ORT 1.18); it no longer allocates state with raw
`cudaMalloc/cudaFree`. Reuse follows the CUDA provider's allocator configuration;
no additional option is needed in `onnx_infer.py`.

The focused correctness checks can be run explicitly after building the plugin:

```bash
XMETAI_TEST_PLUGIN=/absolute/path/to/xmetai_onnx_plugins.so python -m unittest discover -s tests
```

These checks require NumPy, ONNX, ONNX Runtime GPU and a CUDA device. They cover
key-dimension dispatch boundaries, partial warps, initial states, normalization,
multiple batches and overlapping calls on the same session. They do not run a
weather model or measure inference performance.

## Naming

The recommended sub-library name is `xmetai-onnx-plugins`.

This keeps the repository/library name aligned with the existing XMetAI
sub-libraries such as `xmetai-core` and `xmetai-skills`, while the Python
extension module keeps the import/build name `xmetai_onnx_plugins`.

## Build Requirements

- Python with `setuptools` and `packaging`
- CUDA 12.0 or newer
- ONNX Runtime GPU development package with headers and libraries
- A supported NVIDIA GPU architecture for the configured `nvcc` flags

The current build script defaults to the ONNX Runtime package under:

```text
/workspace/github/onnxruntime-linux-x64-gpu-1.xx.x/
```

Set `ORT_PACKAGE_DIR`, `ORT_INCLUDE_DIR`, or `ORT_LIB_DIR` if your local ONNX
Runtime installation uses a different path. `XMETAI_ONNX_PLUGINS_VERSION` can
be used to override the package version during builds. The ONNX Runtime version
is read from `ORT_PACKAGE_DIR/VERSION_NUMBER` or the package directory name; set
`ORT_VERSION` explicitly when neither contains it.

## Build

From this directory:

```bash
python setup.py build_ext --inplace
```

The shared library name records the ONNX Runtime and CUDA toolkit versions and
does not contain a Python ABI tag, for example:

```text
xmetai_onnx_plugins.ort1_27.cuda13.so
```

For packaging or installation:

```bash
pip install .
```

## License

This project is licensed under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE).

Parts of the selective scan implementation are derived from or adapted with
reference to the Mamba project by Tri Dao and Albert Gu:

- Repository: https://github.com/state-spaces/mamba
- License: Apache License, Version 2.0

The attribution is recorded in [`NOTICE`](NOTICE). Source files derived from
Mamba retain the original copyright notice where applicable.

Parts of the Gated DeltaNet-2 implementation are adapted with reference to the
Flash Linear Attention project:

- Repository: https://github.com/fla-org/flash-linear-attention
- License: MIT
