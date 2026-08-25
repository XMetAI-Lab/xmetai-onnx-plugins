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
