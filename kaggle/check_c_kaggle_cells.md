# check_c.py on Kaggle

New Kaggle notebook, Accelerator **GPU T4 x2**, Internet **on**. **Add Input**: your `rgbx-data` dataset (the data zip) and a new dataset `rgbx-weights` (see below). Run the 3 cells with **Run All** (about 8 minutes; no need for Save Version).

## Weights dataset (once)
In Google Drive open `MyDrive/rgbx/runs/`, right-click the folder **`full_rgbd_s0`** → Download, then the same for **`full_rgb_s0`**. Each comes as a zip of about 170 MB. On Kaggle: Datasets → New Dataset → upload both zips, keep it private, name it `rgbx-weights`.

## Cell 1
```python
# Cell 1: check the GPU, get the code
import os, glob, subprocess, torch
def sh(cmd):
    print('$', cmd, flush=True); return subprocess.run(cmd, shell=True).returncode

print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
assert torch.cuda.is_available(), 'Turn on the GPU: Settings > Accelerator > GPU T4 x2'
torch.ones(1).cuda().sum().item()   # an error here means this GPU is not supported: choose GPU T4 x2

CODE = '/tmp/code'
sh(f'rm -rf {CODE} && git clone -q https://github.com/talha-ansarii/btech-project.git {CODE}')
sh(f'cd {CODE} && git log --oneline -1')
sh(f'pip -q install -r {CODE}/requirements.txt')
```

## Cell 2
```python
# Cell 2: find the data and the two full-data model weights
INPUT = '/kaggle/input'
def find(pattern):
    hits = glob.glob(f'{INPUT}/**/{pattern}', recursive=True) + glob.glob(f'/tmp/unz/**/{pattern}', recursive=True)
    if not hits:   # Kaggle sometimes keeps uploaded zips as zips: unpack them once and look again
        for z in glob.glob(f'{INPUT}/**/*.zip', recursive=True):
            sh(f'mkdir -p /tmp/unz && unzip -q -o "{z}" -d /tmp/unz')
        hits = glob.glob(f'/tmp/unz/**/{pattern}', recursive=True)
    assert hits, f'{pattern} not found: add the dataset that holds it with "Add Input"'
    return os.path.dirname(hits[0])

DATA = find('meta.json')
RGBD_RUN = find('full_rgbd_s0/final.pt')   # folder that holds final.pt of the RGB-D model
RGB_RUN = find('full_rgb_s0/final.pt')
RUNS = '/tmp/runs'                          # check_c.py wants <runs>/full_rgbd_s0/final.pt
sh(f'rm -rf {RUNS} && mkdir -p {RUNS} && ln -s {RGBD_RUN} {RUNS}/full_rgbd_s0 && ln -s {RGB_RUN} {RUNS}/full_rgb_s0')
CACHE = '/kaggle/working/cache'
print(len(os.listdir(f'{DATA}/train/rgb')), len(os.listdir(f'{DATA}/test/rgb')), '(expect 795 654)')
sh(f'ls -la {RUNS}/full_rgbd_s0/ {RUNS}/full_rgb_s0/')
```

## Cell 3
```python
# Cell 3: C for the training pool (thresholds only, ~1 min), then the check on the 654 test images (~5 min)
sh(f'python {CODE}/complementarity.py --data {DATA} --split train --out {CACHE}/labelfree_train.npz --no-features')
sh(f'python {CODE}/check_c.py --data {DATA} --runs {RUNS} --cache {CACHE}')
```

## Paste back
The 6 lines printed at the end of Cell 3, starting at "654 test images, mean per-image gain from depth ...".
