import os
import shutil
import subprocess
import sys

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(project_dir, "temp_basicsr")
    
    # 1. Clean up existing temp directory if it exists
    if os.path.exists(temp_dir):
        print(f"Cleaning up existing {temp_dir}...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    # 2. Clone BasicSR repository
    print("Cloning BasicSR repository...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/xinntao/BasicSR.git", temp_dir], check=True)
    
    # 3. Patch setup.py
    setup_py_path = os.path.join(temp_dir, "setup.py")
    print(f"Patching {setup_py_path}...")
    with open(setup_py_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Replace the locals() access in get_version()
    old_func = """def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']"""

    new_func = """def get_version():
    with open(version_file, 'r') as f:
        g = {}
        exec(compile(f.read(), version_file, 'exec'), g)
    return g['__version__']"""

    if old_func in content:
        content = content.replace(old_func, new_func)
    else:
        # Alternative search in case spacing is slightly different
        target = "return locals()['__version__']"
        replacement = "g = {}; exec(compile(open(version_file).read(), version_file, 'exec'), g); return g['__version__']"
        # We find get_version and patch it
        content = content.replace(target, "g = {}; exec(compile(open(version_file).read(), version_file, 'exec'), g); return g['__version__']")

    with open(setup_py_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    # 4. Install using the current python executable's pip
    python_exe = sys.executable
    print(f"Installing BasicSR using {python_exe}...")
    # We set BASICSR_EXT=True to compile if needed, or leave it default
    # BasicSR by default tries to compile CUDA extensions. To prevent build failures on systems without CUDA compiler,
    # we can disable building extensions by not setting BASICSR_EXT=True or explicitly setting BASICSR_EXT=False if needed.
    # Usually, leaving it default installs the pure Python code + basic setup.
    env = os.environ.copy()
    env["BASICSR_EXT"] = "False"  # Force pure Python installation to avoid needing a C++ compiler
    subprocess.run([python_exe, "-m", "pip", "install", temp_dir], env=env, check=True)
    
    # 5. Clean up
    print("Cleaning up temp directory...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("BasicSR installed successfully!")

if __name__ == "__main__":
    main()
