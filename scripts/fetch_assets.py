#!/usr/bin/env python3
"""usage: fetch_assets.py [--tag v0.1.0] [--only checkpoints,heldout,forecast,video] [--local DIR] [--dest assets] [--keep-tars]"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tarfile, tempfile, urllib.request

REPO = 'KosarajuBioAI/HyperGraph-Clustering'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# part -> (tar, prefix of the member paths it holds)
TARS = {'checkpoints': [('checkpoints.tar', 'checkpoints/')],
        'heldout': [('rollouts_heldout_7piezo.tar', 'rollouts/heldout/h7_'),
                    ('rollouts_heldout_14piezo.tar', 'rollouts/heldout/h14_')],
        'forecast': [('rollouts_forecast.tar', 'rollouts/forecast/')],
        'video': [('video_rollouts.tar', 'videos/')]}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def read_sums(p):
    return {l.split(None, 1)[1].strip(): l.split()[0] for l in open(p) if l.strip()}


def api(url, token, accept):
    return urllib.request.Request(url, headers={'Authorization': f'Bearer {token}', 'Accept': accept,
                                                'X-GitHub-Api-Version': '2022-11-28'})


def download(tag, name, tmp):
    if shutil.which('gh'):
        subprocess.run(['gh', 'release', 'download', tag, '-R', REPO, '-p', name, '-D', tmp, '--clobber'], check=True)
        return os.path.join(tmp, name)
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        sys.exit('need gh on PATH or GITHUB_TOKEN')
    with urllib.request.urlopen(api(f'https://api.github.com/repos/{REPO}/releases/tags/{tag}', token,
                                    'application/vnd.github+json')) as r:
        assets = json.load(r)['assets']
    a = next((a for a in assets if a['name'] == name), None)
    if a is None:
        sys.exit(f'{name} not in release {tag}')
    out = os.path.join(tmp, name)
    with urllib.request.urlopen(api(a['url'], token, 'application/octet-stream')) as r, open(out, 'wb') as f:
        shutil.copyfileobj(r, f, 1 << 24)
    return out


def safe(member, dest):
    n = member.name
    if os.path.isabs(n) or '..' in n.split('/') or not (member.isreg() or member.isdir()):
        raise tarfile.TarError(f'refusing member {n}')
    return member


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='v0.1.0')
    ap.add_argument('--only', default=','.join(TARS))
    ap.add_argument('--local')
    ap.add_argument('--dest', default='assets')
    ap.add_argument('--keep-tars', action='store_true')
    a = ap.parse_args()
    parts = a.only.split(',')
    bad = [p for p in parts if p not in TARS]
    if bad:
        sys.exit(f'unknown part {bad}; choose from {",".join(TARS)}')
    man = os.path.join(ROOT, 'assets_manifest')
    if not os.path.exists(os.path.join(man, 'SHA256SUMS')) and a.local:
        man = a.local
    tar_sums, file_sums = read_sums(os.path.join(man, 'SHA256SUMS')), read_sums(os.path.join(man, 'assets_files.sha256'))
    os.makedirs(a.dest, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='hgc_assets_')
    try:
        for part in parts:
            for name, pre in TARS[part]:
                want = {f: h for f, h in file_sums.items() if f.startswith(pre)}
                if want and all(os.path.isfile(os.path.join(a.dest, f)) and sha(os.path.join(a.dest, f)) == h
                                for f, h in want.items()):
                    print(f'{name}: already present')
                    continue
                if a.local:
                    src = os.path.join(a.local, name)
                    if not os.path.isfile(src):
                        sys.exit(f'{src} missing')
                    t = os.path.join(tmp, name)
                    shutil.copyfile(src, t)
                else:
                    t = download(a.tag, name, tmp)
                if sha(t) != tar_sums[name]:
                    os.remove(t)
                    sys.exit(f'sha256 mismatch: {name}')
                with tarfile.open(t) as tf:
                    tf.extractall(a.dest, filter=safe)
                for f, h in want.items():
                    if not os.path.isfile(os.path.join(a.dest, f)) or sha(os.path.join(a.dest, f)) != h:
                        sys.exit(f'sha256 mismatch after extract: {f}')
                print(f'{name}: ok ({len(want)} files)')
                if a.keep_tars:
                    shutil.move(t, os.path.join(a.dest, name))
                else:
                    os.remove(t)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
