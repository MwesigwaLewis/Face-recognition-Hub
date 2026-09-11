import subprocess, sys, traceback

print('=' * 62)
print('LEWISCRYPT HUB — LEGACY CUDA RUNTIME TEST')
print('=' * 62)


def run(cmd):
    print('\n> ' + ' '.join(cmd))
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.stdout.strip(): print(p.stdout.strip())
    if p.stderr.strip(): print(p.stderr.strip())
    return p

p = run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'])
if p.returncode != 0:
    print('\nRESULT: NVIDIA-SMI could not access the GPU.')
    sys.exit(2)

print('\nInstalling the legacy CUDA 12.1 ONNX Runtime stack...')
run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'onnxruntime', 'onnxruntime-gpu'])
p = run([sys.executable, '-m', 'pip', 'install', '--upgrade',
         'onnxruntime-gpu==1.18.0',
         'nvidia-cuda-runtime-cu12==12.1.105',
         'nvidia-cublas-cu12==12.1.3.1',
         'nvidia-cudnn-cu12==8.9.7.29'])
if p.returncode != 0:
    print('\nRESULT: Legacy CUDA runtime installation FAILED.')
    sys.exit(3)

print('\nTesting ONNX Runtime providers...')
try:
    import onnxruntime as ort
    print('ONNX Runtime:', ort.__version__)
    print('Providers:', ort.get_available_providers())
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        print('\nRESULT: CUDAExecutionProvider is NOT available.')
        sys.exit(4)
except Exception:
    traceback.print_exc()
    sys.exit(5)

print('\nTesting actual InsightFace detector inference on CUDA...')
try:
    import numpy as np
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.55)
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    faces = app.get(frame)
    print('Inference completed.')
    print('Detections on blank frame:', len(faces))
    print('\nRESULT: CUDA INFERENCE WORKS.')
    print('Lewiscrypt can use the GPU runtime path.')
    sys.exit(0)
except Exception as e:
    print('\nRESULT: CUDA provider is visible, but actual InsightFace CUDA inference FAILED.')
    print(type(e).__name__ + ':', e)
    traceback.print_exc()
    sys.exit(6)
