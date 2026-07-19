import os
import shutil
import subprocess
import sys
import glob

def main():
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(tools_dir)
    temp_dir = os.path.join(tools_dir, "temp_basicsr_build")
    wheels_dir = os.path.join(project_dir, "wheels")
    
    # 1. Ensure target directory exists and is clean
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(wheels_dir, exist_ok=True)
    
    # 2. Clone BasicSR
    print("Cloning BasicSR from GitHub...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/xinntao/BasicSR.git", temp_dir], check=True)
    
    # 3. Patch setup.py to support Python 3.11+
    setup_py_path = os.path.join(temp_dir, "setup.py")
    print("Patching setup.py...")
    with open(setup_py_path, "r", encoding="utf-8") as f:
        content = f.read()
        
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
        target = "return locals()['__version__']"
        replacement = "g = {}; exec(compile(open(version_file).read(), version_file, 'exec'), g); return g['__version__']"
        content = content.replace(target, replacement)

    with open(setup_py_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    # 4. Build the platform-independent wheel
    python_exe = sys.executable
    print("Installing build dependencies (wheel)...")
    subprocess.run([python_exe, "-m", "pip", "install", "wheel"], check=True)
    
    print("Building BasicSR wheel...")
    env = os.environ.copy()
    env["BASICSR_EXT"] = "False"  # Force pure python platform-independent wheel
    subprocess.run([python_exe, "setup.py", "bdist_wheel"], cwd=temp_dir, env=env, check=True)
    
    # 5. Base64 encode the built wheel and save it in tools/ directory
    import base64
    wheel_files = glob.glob(os.path.join(temp_dir, "dist", "*.whl"))
    if not wheel_files:
        raise FileNotFoundError("No built wheel file found in dist directory.")
        
    whl_path = wheel_files[0]
    b64_dest = os.path.join(tools_dir, "basicsr-1.4.2-py3-none-any.whl.b64")
    print(f"Base64 encoding {whl_path} into {b64_dest}...")
    with open(whl_path, "rb") as f_in:
        encoded_data = base64.b64encode(f_in.read())
        
    with open(b64_dest, "wb") as f_out:
        f_out.write(encoded_data)
        
    # 6. Cleanup
    print("Cleaning up temporary directories...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Patched BasicSR wheel base64 encoded and built successfully!")

if __name__ == "__main__":
    main()
