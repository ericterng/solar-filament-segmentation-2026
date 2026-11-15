"""Pin one physical GPU by UUID (chosen at start), never by index.

Shared-host version of solar.yolo_device (which hardcodes the physical
index 1 of one host). The UUID comes from the caller; the module checks it exists, sets
CUDA_VISIBLE_DEVICES before CUDA initialises, and verifies through nvidia-smi
that this PID really runs on that UUID and nowhere else.
"""
import os
import re
import subprocess

UUID_RE = re.compile(r'^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


def physical_index(uuid: str) -> int:
    """Physical nvidia-smi index of `uuid`; raises if the UUID is unknown."""
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    rows = {u.strip(): int(i) for i, u in (line.split(', ', 1) for line in output.strip().splitlines())}
    if uuid not in rows:
        raise RuntimeError(f'GPU UUID not present on this host: {uuid}')
    return rows[uuid]


def pin_uuid(uuid: str) -> int:
    """Validate `uuid`, restrict CUDA to it before torch initialises CUDA; return its physical index."""
    if not UUID_RE.match(uuid):
        raise ValueError(f'Not a GPU UUID: {uuid!r}')
    index = physical_index(uuid)
    import torch
    if torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise RuntimeError('CUDA already initialised without the selected UUID')
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    return index


def assert_pinned(uuid: str) -> dict:
    """Check the env string and the real PID -> UUID mapping (allocates 4 bytes on cuda:0)."""
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise RuntimeError('CUDA visibility was overwritten')
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f'Expected exactly one visible GPU, got {torch.cuda.device_count()}')
    torch.empty(1, device=torch.device('cuda:0'))
    torch.cuda.synchronize()
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'],
                                     text=True)
    actual = {line.split(', ', 1)[1].strip() for line in output.strip().splitlines()
              if line.split(', ', 1)[0].strip() == str(os.getpid())}
    if actual != {uuid}:
        raise RuntimeError(f'Physical GPU mismatch for PID {os.getpid()}: {actual}; expected {uuid}')
    return {'pid': os.getpid(), 'physical_index': physical_index(uuid), 'uuid': uuid,
            'verified_by': 'nvidia-smi process mapping'}
