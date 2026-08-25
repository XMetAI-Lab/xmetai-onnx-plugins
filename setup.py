"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-12-29 10:28:52
Description: onnx-plugins for xmetai custom ops.
Copyright (c) 2025 by XMetAI/douzesheng.
Licensed under the Apache License, Version 2.0.
"""

import warnings
import os
import re
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

from setuptools import Extension
from setuptools.command.build_ext import build_ext

with open(Path(__file__).resolve().parent / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "xmetai-onnx-plugins"
EXTENSION_NAME = "xmetai_onnx_plugins"
PACKAGE_VERSION = os.environ.get("XMETAI_ONNX_PLUGINS_VERSION", "0.1.0")

ORT_PACKAGE_DIR = os.environ.get("ORT_PACKAGE_DIR", "/workspace/github/onnxruntime-linux-x64-gpu-1.23.2").rstrip("/")
ORT_INCLUDE_DIR = os.environ.get("ORT_INCLUDE_DIR", os.path.join(ORT_PACKAGE_DIR, "include"))
ORT_LIB_DIR = os.environ.get("ORT_LIB_DIR", os.path.join(ORT_PACKAGE_DIR, "lib"))
if not os.path.exists(ORT_INCLUDE_DIR):
    raise RuntimeError(f"Cannot locate ONNX Runtime headers under {ORT_INCLUDE_DIR}")
if not os.path.exists(ORT_LIB_DIR):
    raise RuntimeError(f"Cannot locate ONNX Runtime libraries under {ORT_LIB_DIR}")


def get_ort_version():
    configured_version = os.environ.get("ORT_VERSION")
    if configured_version:
        return str(Version(configured_version))

    version_file = Path(ORT_PACKAGE_DIR) / "VERSION_NUMBER"
    if version_file.is_file():
        return str(Version(version_file.read_text(encoding="utf-8").strip()))

    match = re.search(r"(?<!\d)(\d+\.\d+\.\d+(?:[a-z0-9.-]*)?)$", Path(ORT_PACKAGE_DIR).name, re.IGNORECASE)
    if match:
        return str(Version(match.group(1)))

    raise RuntimeError(
        f"Cannot determine the ONNX Runtime version from {ORT_PACKAGE_DIR}. "
        "Set ORT_VERSION explicitly."
    )


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_ver = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_ver


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


cmdclass: dict[str, type] = {}
ext_modules = []


def find_cuda_home():
    paths = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
    ]
    for p in paths:
        if p and os.path.exists(p):
            return p
    try:
        nvcc_path = subprocess.check_output(["which", "nvcc"], universal_newlines=True).strip()
        if nvcc_path:
            return os.path.dirname(os.path.dirname(nvcc_path))
    except Exception:
        pass
    return None


CUDA_HOME = find_cuda_home()
if CUDA_HOME is None:
    check_if_cuda_home_none(PACKAGE_NAME)
    bare_metal_version = Version("0")
else:
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("11.6"):
        raise RuntimeError(f"{PACKAGE_NAME} is only supported on CUDA 11.6 and above.  " "Note: make sure nvcc has a supported version by running nvcc -V.")

ORT_VERSION = get_ort_version()
ort_version = Version(ORT_VERSION)
PLUGIN_FILENAME = f"{EXTENSION_NAME}.ort{ort_version.major}_{ort_version.minor}.cuda{bare_metal_version.major}.so"

cuda_arches = ["80", "86", "89", "90"]
if bare_metal_version < Version("13.0"):
    cuda_arches.insert(0, "70")
if bare_metal_version >= Version("12.8"):
    cuda_arches.append("120")

cc_flag = []
for arch in cuda_arches:
    cc_flag.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": append_nvcc_threads(
        [
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
        ]
        + cc_flag
    ),
}

CUDA_SOURCES = [
    "csrc/selective_scan/selective_scan_fwd_fp32.cu",
    "csrc/selective_scan/selective_scan_fwd_fp16.cu",
    # "csrc/selective_scan/selective_scan_fwd_bf16.cu",
    "csrc/mamba3_siso/mamba3_siso_fwd.cu",
    "csrc/gated_deltanet2/gated_deltanet2_fwd.cu",
]

ext = Extension(
    name=EXTENSION_NAME,
    sources=[
        "csrc/register_op.cpp",
        "csrc/softmax_cpu/softmax_cpu_ort.cpp",
        "csrc/selective_scan/selective_scan_fn_ort.cpp",
        "csrc/mamba3_siso/mamba3_siso_fn_ort.cpp",
        "csrc/gated_deltanet2/gated_deltanet2_fn_ort.cpp",
    ],
    include_dirs=[
        str(Path(this_dir) / "csrc" / "selective_scan"),
        str(Path(this_dir) / "csrc" / "mamba3_siso"),
        str(Path(this_dir) / "csrc" / "gated_deltanet2"),
        str(Path(this_dir) / "csrc" / "softmax_cpu"),
        str(Path(this_dir) / "csrc" / "utils"),
        str(ORT_INCLUDE_DIR),
        os.path.join(CUDA_HOME, "include") if CUDA_HOME else "/usr/local/cuda/include",
    ],
    libraries=["onnxruntime", "cudart"],
    library_dirs=[
        str(ORT_LIB_DIR),
        os.path.join(CUDA_HOME, "lib64") if CUDA_HOME else "/usr/local/cuda/lib64",
    ],
    runtime_library_dirs=[
        str(ORT_LIB_DIR),
        os.path.join(CUDA_HOME, "lib64") if CUDA_HOME else "/usr/local/cuda/lib64",
    ],
    extra_compile_args=extra_compile_args,
)
ext.cuda_sources = CUDA_SOURCES
ext_modules.append(ext)


class CudaBuildExtension(build_ext):
    def get_ext_filename(self, ext_name):
        if ext_name == EXTENSION_NAME:
            return PLUGIN_FILENAME
        return super().get_ext_filename(ext_name)

    def build_extension(self, ext):
        # Separate CUDA and C/C++ sources
        sources = list(ext.sources)
        cu_sources = list(getattr(ext, "cuda_sources", []))
        cpp_sources = sources

        objects = []

        # Compile C/C++ sources with default compiler
        if cpp_sources:
            objects.extend(
                self.compiler.compile(
                    cpp_sources,
                    output_dir=self.build_temp,
                    include_dirs=ext.include_dirs,
                    debug=self.debug,
                    extra_postargs=ext.extra_compile_args.get("cxx", []),
                    depends=ext.depends,
                )
            )

        # Compile CUDA sources with nvcc
        if cu_sources:
            if CUDA_HOME is None:
                raise RuntimeError("nvcc not found and CUDA_HOME is None; cannot compile CUDA sources")
            nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
            os.makedirs(self.build_temp, exist_ok=True)
            for src in cu_sources:
                # Object file path (avoid distutils .cu unknown type)
                base = os.path.splitext(os.path.basename(src))[0]
                obj = os.path.join(self.build_temp, base + ".o")
                cmd = [
                    nvcc,
                    "-c",
                    src,
                    "-o",
                    obj,
                    "-Xcompiler",
                    "-fPIC",
                ]
                # Add include dirs
                for inc in ext.include_dirs:
                    cmd.extend(["-I", inc])
                # Add nvcc extra args
                cmd.extend(ext.extra_compile_args.get("nvcc", []))
                self.spawn(cmd)
                objects.append(obj)

        # Link all objects into the extension
        output_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Linker args
        extra_link_args = []
        runtime_library_dirs = getattr(ext, "runtime_library_dirs", [])
        # Ensure rpath
        for rdir in runtime_library_dirs:
            extra_link_args.extend(["-Wl,-rpath," + rdir])
        self.compiler.link_shared_object(
            objects,
            output_path,
            libraries=ext.libraries,
            library_dirs=ext.library_dirs,
            runtime_library_dirs=runtime_library_dirs,
            extra_postargs=extra_link_args,
        )


def get_package_version():
    return PACKAGE_VERSION


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
        )
    ),
    author="DouZesheng",
    author_email="douzsh@gmail.com",
    description="Onnx Plugins for XMetAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    python_requires=">=3.9",
    install_requires=[
        "packaging",
        "ninja",
    ],
    cmdclass={"build_ext": CudaBuildExtension},
)
