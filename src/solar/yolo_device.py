"""Pin the selected physical GPU by UUID, never let a library remap it."""
import os
import subprocess

import torch


def authorized_uuid() -> str:
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    rows = dict(line.strip().split(', ', 1) for line in output.strip().splitlines())
    return rows['1']


def pin_gpu_one() -> torch.device:
    """Resolve physical index1 before CUDA init; use a device object downstream."""
    uuid = authorized_uuid()
    if torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise RuntimeError('CUDA already initialized without the selected UUID')
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    return torch.device('cuda:0')


def assert_gpu_one() -> dict:
    """Check actual PID ownership through NVML/nvidia-smi, not device labels."""
    expected = authorized_uuid()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected:
        raise RuntimeError('CUDA visibility was overwritten')
    allocation = torch.empty(1, device=torch.device('cuda:0'))
    torch.cuda.synchronize()
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    actual = {line.split(', ', 1)[1].strip() for line in output.strip().splitlines()
              if line.split(', ', 1)[0].strip() == str(os.getpid())}
    if actual != {expected}:
        raise RuntimeError(f'Physical GPU mismatch for PID {os.getpid()}: {actual}; expected {expected}')
    return {'pid': os.getpid(), 'physical_index': 1, 'uuid': expected, 'verified_by': 'nvidia-smi process mapping'}
