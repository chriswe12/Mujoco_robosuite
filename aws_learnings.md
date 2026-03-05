# AWS Learnings

## Context

- Instance: `g6.xlarge`
- AMI ID: `ami-07f38ba1a2d825796`
- AMI name: `ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-20251212`
- Root disk observed on first boot: `8G`
- Extra local NVMe observed on instance: `/dev/nvme1n1` (`~232.8G`) mounted manually at `/mnt/work`

## What We Learned

### 1. Generic Ubuntu GPU instances do not guarantee NVIDIA drivers

- On this AMI, `nvidia-smi` was not installed and `torch.cuda.is_available()` was `False`.
- The fix was to install the NVIDIA driver on the instance:

```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers --gpgpu install
sudo reboot
```

- After reboot, the scratch disk mount had to be recreated:

```bash
sudo mount /dev/nvme1n1 /mnt/work
sudo chown ubuntu:ubuntu /mnt/work
```

### 2. The default root disk was too small for GPU Python packages

- `pip install` for CUDA-enabled PyTorch failed with `OSError: [Errno 28] No space left on device`.
- The default `8G` root volume is too small for this workflow.
- The large local NVMe disk worked as scratch space, but that was a workaround, not the clean setup.

### 3. `/mnt/work` on `/dev/nvme1n1` is useful but ephemeral

- Using the local NVMe disk for the repo, venv, temp files, and pip cache made installation practical.
- Files under `/mnt/work` should be treated as temporary.
- SSH disconnect is fine.
- Reboot is fine, but the mount must be recreated.
- Stop/terminate should be assumed to destroy `/mnt/work` contents.

### 4. The repo-wide `requirements.txt` is too broad for remote Humanoid training

- `requirements.txt` includes `robosuite`.
- `robosuite` pulled in `evdev`, which then failed to build because kernel headers were missing.
- For `scripts/train_deploy_humanoid.py`, `robosuite` is not needed.
- The working remote install path was to install only the packages needed for Humanoid training:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy mujoco gymnasium stable-baselines3 imageio
```

### 5. `g6.xlarge` has enough GPU, but not a lot of CPU

- `g6.xlarge` is usable for this training flow.
- It is not a good match for aggressive vectorization.
- `--n-envs 8` is too high for this machine.
- A practical starting point is `--n-envs 2` or `--n-envs 4`.

### 6. `SubprocVecEnv` start method mattered on AWS

- The script default was changed to `spawn` earlier to avoid a local sandbox issue.
- On a normal Linux AWS instance, `fork` was the better choice for this workload.
- The reliable training command on AWS was:

```bash
python scripts/train_deploy_humanoid.py \
  --mode train \
  --device cuda \
  --start-method fork \
  --n-envs 2 \
  --n-steps 1024 \
  --batch-size 1024 \
  --timesteps 1000000
```

## What Worked

### Scratch disk setup

```bash
sudo mkfs.ext4 -F /dev/nvme1n1
sudo mkdir -p /mnt/work
sudo mount /dev/nvme1n1 /mnt/work
sudo chown ubuntu:ubuntu /mnt/work
```

### Working-directory setup on scratch storage

```bash
cp -a ~/Mujoco_robosuite /mnt/work/
cd /mnt/work/Mujoco_robosuite
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Temp/cache redirection

```bash
mkdir -p /mnt/work/tmp /mnt/work/pip-cache /mnt/work/.cache
export TMPDIR=/mnt/work/tmp
export PIP_CACHE_DIR=/mnt/work/pip-cache
export XDG_CACHE_HOME=/mnt/work/.cache
export PIP_NO_CACHE_DIR=1
```

### Sanity checks

```bash
nvidia-smi

python - <<'PY'
import torch, mujoco, gymnasium, stable_baselines3
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('mujoco', mujoco.__version__)
print('gymnasium', gymnasium.__version__)
print('sb3', stable_baselines3.__version__)
PY
```

### Artifact copy-out

- The policy and normalization stats were saved at:
  - `/mnt/work/Mujoco_robosuite/artifacts/humanoid_ppo.zip`
  - `/mnt/work/Mujoco_robosuite/artifacts/humanoid_vecnormalize.pkl`
- These should be copied off the instance before stop/terminate.

## Suggestions For Next Time

### Better volume strategy

1. The clean default is to launch the instance with a larger root EBS volume from the start.
2. For this workflow, a root volume in the `100-150G` range is simpler than relying on scratch NVMe.
3. If persistence matters, prefer EBS over instance store for the main workspace.
4. If a separate workspace volume is desired, attach an additional EBS volume at launch and mount it at something like `/workspace` or `/data`.
5. Use instance store only when the speed benefit is worth the operational tradeoff:
   - it is fast
   - it is convenient for temporary scratch data
   - it is not the right place to keep the only copy of models or training outputs

### Better automation strategy

1. Use an EC2 Launch Template so the AMI, instance type, block device mappings, security group, and user data are fixed and repeatable.
2. Put bootstrapping in user data / cloud-init so the machine self-configures on first boot.
3. Use user data to automate:
   - driver install
   - volume mount
   - repo checkout
   - venv creation
   - dependency install
4. If this workflow will be reused, bake a custom AMI after one machine is configured correctly.
5. If the setup needs to become durable and repeatable across teams, move the bootstrap into EC2 Image Builder, Terraform, or both.

### Repo/documentation suggestions

1. Add a dedicated `requirements-humanoid.txt` or `requirements-remote-humanoid.txt` that excludes `robosuite`.
2. Add an `AWS_REMOTE_TRAINING.md` with:
   - recommended instance sizes
   - AMI guidance
   - scratch-disk setup
   - driver install steps
   - exact `scp` commands for artifacts
3. Add a note that `g6.xlarge` should start at `--n-envs 2` or `4`, not `8`.
4. Add a note that `/mnt/work` on instance store is ephemeral and should not be the only place artifacts live.
5. Add a remote-training section that explicitly says `--mode train` on headless servers and `--mode deploy` locally.

### Script suggestions

1. Add a short warning when `n_envs` is obviously high relative to `os.cpu_count()`.
2. Add an `--artifacts-dir` shortcut for remote runs.
3. Add a `--smoke-test` preset for quick validation on new machines.
4. Revisit the default multiprocessing start method:
   - `spawn` was needed for the local sandbox
   - `fork` behaved better on real Linux AWS instances
   - this likely needs either a platform-specific default or a clearer README note

### Infrastructure suggestions

1. Prefer a larger root EBS volume from the start, even if scratch NVMe is available.
2. If persistence matters, use a second EBS volume for the workspace instead of relying on instance store.
3. Prefer an AMI or bootstrap path that guarantees NVIDIA drivers are already installed.
4. Copy artifacts to local disk or S3 before stopping the instance.
5. If using scratch NVMe, automate the mount in bootstrapping instead of doing it manually.
6. Store the known-good EC2 launch configuration as a Launch Template instead of rebuilding it manually every time.

### Concrete preferred setup next time

1. Start from a GPU-ready AMI or an AMI plus user-data bootstrap that installs the NVIDIA driver.
2. Launch with a larger root EBS volume immediately.
3. If a separate workspace is useful, attach persistent EBS and mount it automatically at boot.
4. Use a launch template to encode:
   - AMI
   - instance type
   - root volume size
   - extra EBS volume
   - user data bootstrap
5. Use user data to:
   - install drivers
   - mount the workspace
   - clone the repo
   - create the venv
   - install the minimal Humanoid requirements
6. Use a custom AMI once the setup is stable to avoid redoing driver/bootstrap steps on every instance.
