import os
import sys
import subprocess
import time

# Ensure project root is on sys.path
tools_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(tools_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

def run_test_script(script_name):
    script_path = os.path.join(tools_dir, script_name)
    print(f"\n==================================================")
    print(f"   RUNNING VERIFICATION SUITE: {script_name}")
    print(f"==================================================")
    
    t0 = time.time()
    cmd = [sys.executable, script_path]
    res = subprocess.run(cmd, cwd=project_dir)
    t1 = time.time()
    
    if res.returncode == 0:
        print(f"[OK] {script_name} PASSED in {t1 - t0:.2f}s (exit code 0)")
        return True
    else:
        print(f"[FAILED] {script_name} FAILED with exit code {res.returncode}")
        return False

def main():
    print("==================================================")
    print("   PROJECT AI ENHANCER MASTER VERIFICATION SUITE  ")
    print("==================================================")
    
    test_scripts = [
        "test_evaluation_workflow.py",
        "test_pipeline.py",
        "test_ab_ui_pipeline.py",
        "test_dataset_loader.py",
        "test_stage2_config.py"
    ]
    
    results = {}
    all_passed = True
    
    for script in test_scripts:
        passed = run_test_script(script)
        results[script] = passed
        if not passed:
            all_passed = False
            
    print("\n==================================================")
    print("             SUMMARY TEST REPORT CARD             ")
    print("==================================================")
    for script, passed in results.items():
        status_str = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"  - {script:<30}: {status_str}")
        
    print("==================================================")
    if all_passed:
        print("ALL VERIFICATION SUITES PASSED WITH EXIT CODE 0!")
        sys.exit(0)
    else:
        print("ONE OR MORE VERIFICATION SUITES FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
