import datetime
import secrets
import subprocess
import time
from collections import deque
from itertools import product
from typing import List

rand_gen = secrets.SystemRandom()

N = 10
ngpu = 4
nproc_per_gpu = 4


beta1s = [0.1, 0.3, 0.5, 0.7, 0.9]
beta2s = [0.1, 0.3, 0.5, 0.7, 0.9]
experiments = deque()
for b1, b2 in product(beta1s, beta2s):
    experiments.extend((b1, b2, rand_gen.randint(0, 1000000)) for _ in range(N))


def get_cmd(beta1, beta2, seed, device):
    return f"python main.py --dataset acm --heco_drop --device {device} --beta1 {beta1} --beta2 {beta2} --seed {seed}"


procs: List[subprocess.Popen] = []
for i in range(ngpu):
    for _ in range(nproc_per_gpu):
        if len(experiments) == 0:
            continue
        b1, b2, seed = experiments.popleft()
        cmd = get_cmd(b1, b2, seed, f"cuda:{i}")
        print(datetime.datetime.now(), cmd)
        procs.append(subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL))


while True:
    for i, proc in enumerate(procs):
        if proc.poll() is not None and len(experiments) > 0:
            b1, b2, seed = experiments.popleft()
            cmd = get_cmd(b1, b2, seed, f"cuda:{i//nproc_per_gpu}")
            print(datetime.datetime.now(), cmd)
            procs[i] = subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL)

    if len(experiments) == 0:
        for p in procs:
            p.wait()
        break

    time.sleep(60)
