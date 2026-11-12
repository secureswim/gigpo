import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo = "Qwen/Qwen2.5-1.5B-Instruct"
info = HfApi().model_info(repo, files_metadata=True)
total = sum(f.size or 0 for f in info.siblings)
print(json.dumps({"repo": repo, "revision": info.sha, "bytes": total}), flush=True)
root = Path('/home/vansh23580/rl-project/gigpo/cluster')
(root / 'model-revision.txt').write_text(info.sha + '\n')
print(snapshot_download(repo, revision=info.sha), flush=True)
