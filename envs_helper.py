import os
from adb_helper import ADB

def setup_mobile_env(host_snpe: str, adbkey_path: str, mobile_arch: str ="arm-android-clang8.0", mobile_folder: str ="/data/local/tmp/snpe"):
    """
    Setup mobile device environment.    
    
    Args:
    host_snpe: 
        Path of SNPE folder (xxx/snpe.1.64xxx/).
    adbkey_path:
        Path of adbkey, usually is "/home/xxx/./android"
    """
    # Setup folders
    adb = ADB(adbkey_path)
    folders = ["bin", "lib", "dsp/lib", "models", "inputs", "outputs"]    
    for f in folders:
        adb.shell(f"mkdir -p {mobile_folder}/{f}")
    
    # Push libraries:
    lib_src = f"{host_snpe}/lib/{mobile_arch}"
    lib_dst = f"{mobile_folder}/lib/"
    for f in os.listdir(lib_src):
        adb.push(f"{lib_src}/{f}", f"{lib_dst}/{f}")
    
    
    # Push dsp files
    dsp_src = f"{host_snpe}/lib/dsp/"
    dsp_dst = f"{mobile_folder}/dsp/lib/"
    for f in os.listdir(dsp_src):
        adb.push(f"{dsp_src}/{f}", f"{dsp_dst}/{f}")

    # Push scripts
    run_src = f"{host_snpe}/bin/{mobile_arch}/snpe-net-run"
    run_dst = f"{mobile_folder}/bin/snpe-net-run"
    adb.push(run_src, run_dst)
     
    # Push scripts
    run_src = f"{host_snpe}/bin/{mobile_arch}/snpe-platform-validator"
    run_dst = f"{mobile_folder}/bin/snpe-platform-validator"
    adb.push(run_src, run_dst)
    
def delete_mobile_envs(adbkey_path:str, mobile_folder :str = "/data/local/tmp/snpe"):
    adb = ADB(adbkey_path)
    adb.shell(f"rm -r {mobile_folder}")
    
    
def setup_host_env():
    os.makedirs("/tmp/snpe/", exist_ok=True)
    os.makedirs("/tmp/snpe/inputs", exist_ok=True)
    os.makedirs("/tmp/snpe/outputs", exist_ok=True)