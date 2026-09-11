import os,sys,ctypes,subprocess
from pathlib import Path
site=Path(sys.prefix)/'Lib'/'site-packages'
print('='*70);print('LEWISCRYPT HUB - CUDA DLL DIAGNOSTIC');print('='*70)
print('Python:',sys.version.split()[0]); print('Venv:',sys.prefix)
print('\n[1] NVIDIA GPU')
try: print(subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True,stderr=subprocess.STDOUT).strip())
except Exception as e: print('FAIL:',e)
print('\n[2] DLL directories')
dirs=[site/'nvidia/cuda_runtime/bin',site/'nvidia/cuda_nvrtc/bin',site/'nvidia/cublas/bin',site/'nvidia/cudnn/bin',site/'onnxruntime/capi']
for d in dirs:
 print('\n',d)
 if d.exists():
  for p in sorted(d.glob('*.dll')): print(' ',p.name)
 else: print(' NOT FOUND')
print('\n[3] DLL load tests')
for d in dirs[:-1]:
 if d.exists():
  try: os.add_dll_directory(str(d))
  except: pass
tests=[('CUDA Runtime',site/'nvidia/cuda_runtime/bin/cudart64_12.dll'),('cuBLAS',site/'nvidia/cublas/bin/cublas64_12.dll'),('cuBLASLt',site/'nvidia/cublas/bin/cublasLt64_12.dll'),('cuDNN',site/'nvidia/cudnn/bin/cudnn64_8.dll')]
for name,p in tests:
 print('\n'+name+':',p)
 if not p.exists(): print(' NOT FOUND'); continue
 try: ctypes.WinDLL(str(p)); print(' LOAD OK')
 except OSError as e: print(' LOAD FAILED:',e); print(' WinError:',getattr(e,'winerror',None))
p=site/'onnxruntime/capi/onnxruntime_providers_cuda.dll'; print('\nONNX Runtime CUDA provider:',p)
try: ctypes.WinDLL(str(p)); print(' LOAD OK')
except OSError as e: print(' LOAD FAILED:',e); print(' WinError:',getattr(e,'winerror',None))
print('\n[4] ONNX Runtime')
try:
 import onnxruntime as ort; print('Version:',ort.__version__); print('Providers:',ort.get_available_providers())
except Exception as e: print('IMPORT FAILED:',repr(e))
print('\n'+'='*70);print('DIAGNOSTIC COMPLETE');print('='*70);input('Press Enter to exit...')
