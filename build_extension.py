import os
import sys
import subprocess
import sysconfig
import pybind11


def build():
    print("=" * 60)
    print(" Compiling Monte Carlo Engine...")
    print("=" * 60)

    
 # 1. Toolchain and include paths
    gxx = r"C:\msys64\ucrt64\bin\g++.exe"
    if not os.path.isfile(gxx):
        gxx = "g++"

    py_include = sysconfig.get_paths()["include"]
    py_libs = os.path.join(sys.base_prefix, "libs")
    py_ver = f"python{sys.version_info.major}{sys.version_info.minor}"
    pybind_include = pybind11.get_include()

    cpp_include = os.path.abspath("cpp/include")
    src_dir = os.path.abspath("cpp/src")
    out_dir = os.path.abspath("risk_engine")
    out_file = os.path.join(out_dir, "_risk_engine.pyd")

    sources = [
        os.path.join(src_dir, "cholesky.cpp"),
        os.path.join(src_dir, "simulation_engine.cpp"),
        os.path.join(src_dir, "bindings.cpp")
    ]

    # Verify source file existence
    for src in sources:
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Source file not found: {src}")

    # 2. Optimized GCC compilation flags
    cmd = [
        gxx,
        "-O3",                      # Maximum compiler optimization
        "-shared",                   # Build dynamic extension module (.pyd)
        "-std=c++17",                # Modern C++17 standard
        f"-I{cpp_include}",          # Engine C++ headers
        f"-I{py_include}",           # Python C-API headers
        f"-I{pybind_include}",       # Pybind11 headers
        *sources,
        "-o", out_file,
        f"-L{py_libs}",
        f"-l{py_ver}"
    ]

    # Prepend MinGW toolchain directory to PATH for internal tool resolution
    env = os.environ.copy()
    msys_bin = r"C:\msys64\ucrt64\bin"
    if os.path.isdir(msys_bin):
        env["PATH"] = msys_bin + os.pathsep + env.get("PATH", "")
    print(f"Executing: {' '.join(cmd)}\n")
    res = subprocess.run(cmd, env=env)
    if res.returncode == 0:
        print("\n" + "=" * 60)
        print(" SUCCESS: C++ extension '_risk_engine.pyd' built successfully!")
        print(f" Output location: {out_file}")
        print("=" * 60)
    else:
        print("\n Compilation FAILED with exit code:", res.returncode)
        sys.exit(res.returncode)
if __name__ == "__main__":
    build()