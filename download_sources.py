"""Download attributed public examples. None are student-captured footage."""
from pathlib import Path
import urllib.request, json, hashlib, concurrent.futures, shutil
ROOT=Path(__file__).resolve().parent;D=ROOT/'data';D.mkdir(exist_ok=True)
COMMIT='4.10.0'
base=f'https://raw.githubusercontent.com/opencv/opencv/{COMMIT}/samples/data/'
items=[(base+'vtest.avi','vtest.avi')]+[(base+f'left{i:02d}.jpg',f'left{i:02d}.jpg') for i in [1,2,3,4,5,6,7,8,9,11,12,13,14]]
items += [(f'https://raw.githubusercontent.com/opencv/opencv/{COMMIT}/LICENSE','OpenCV_LICENSE.txt')]
def get(it):
    u,name=it;p=D/name
    if not p.exists():
        with urllib.request.urlopen(u,timeout=120) as r:p.write_bytes(r.read())
    return dict(file=name,url=u,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: records=list(ex.map(get,items))
v=D/'v_JspVuT6rsLA.mp4'
u='https://raw.githubusercontent.com/Amshaker/Mobile-VideoGPT/c82a4cbc9edcb4eacd6223b4389699f24aca6635/sample_videos/v_JspVuT6rsLA.mp4'
cached=Path('/tmp/week6-mobile-videogpt/sample_videos/v_JspVuT6rsLA.mp4')
if not v.exists() and cached.exists():shutil.copyfile(cached,v)
records.append(get((u,v.name)))
for name in ['SOURCE_LICENSE.txt','SOURCE_README_LICENSE.txt']:
    shutil.copyfile(ROOT.parent/'Week_6_VideoLLM'/name,D/name)
(D/'SOURCES.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
