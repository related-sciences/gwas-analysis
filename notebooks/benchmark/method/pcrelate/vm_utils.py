import platform,socket,re,uuid,json,psutil,logging
import psutil
from pathlib import Path

def get_core_mhz() -> str:
    cpu_info = Path("/proc/cpuinfo")
    cpus_mhz = [l for l in cpu_info.read_text().splitlines() if "cpu MHz" in l]
    cpus_mhz = {x.split(":")[1].strip() for x in cpus_mhz}
    return next(iter(cpus_mhz)) if cpus_mhz else "Unknown"

def get_vm_info() -> str:
    try:
        info={}
        info['platform']=platform.system()
        info['platform-release']=platform.release()
        info['platform-version']=platform.version()
        info['architecture']=platform.machine()
        info['processor']=psutil.cpu_count()
        info['processor_mhz']=get_core_mhz()
        info['ram']=str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
        return json.dumps(info)
    except Exception as e:
        logging.exception(e)