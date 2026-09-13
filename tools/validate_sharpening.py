"""Fixed validation check, disjoint from the six development references."""
import sys, json, argparse, hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline import LocalAIEnhancerPipeline
from tools.evaluate_restoration import BenchmarkEvaluator

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True,help='New directory; existing results are never overwritten.')
args=parser.parse_args()
torch.set_num_threads(8);cv2.setNumThreads(8)
out=args.output
out.mkdir(parents=True,exist_ok=False)
names=(ROOT/'benchmarks/splits/portraits_experiment_v1/validation.txt').read_text().splitlines()
used=set(np.linspace(0,len(names)-1,6,dtype=int))|set(np.linspace(0,len(names)-1,3,dtype=int))
indices=np.random.default_rng(91712).choice([i for i in range(len(names)) if i not in used],12,replace=False)
pipe=LocalAIEnhancerPipeline(device='cpu',lazy_load=True)
report={'status':'running','scope':'12 validation references disjoint from development images, 60 synthetic cases; not production holdout',
 'selected':[names[i] for i in indices], 'strength':.5,'seed':91712,
 'hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('pipeline.py','wink_enhancer.py','app.py','tools/validate_sharpening.py','tools/evaluate_restoration.py')},
 'environment':{'opencv':cv2.__version__,'python':sys.version,'torch':torch.__version__},'samples':[]}
def save(): (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
save()
eval=BenchmarkEvaluator(device='cpu')
try:
 for i in indices:
    ref=cv2.imread(str(ROOT/'models/CodeFormer/datasets/ffhq/ffhq_512'/names[i]))
    ref=cv2.resize(ref,(512,512),interpolation=cv2.INTER_AREA)
    for condition in ('soft','severe','noise','clean','motion'):
        scale=4 if condition=='severe' else (1 if condition=='clean' else 2)
        low=ref
        if condition=='motion':
            kernel=np.zeros((7,7),np.float32);kernel[3]=1/7;low=cv2.filter2D(low,-1,kernel)
        elif condition!='clean':low=cv2.GaussianBlur(low,(0,0),2.2 if condition=='severe' else 1.2)
        low=cv2.resize(low,(512//scale,512//scale),interpolation=cv2.INTER_AREA)
        if condition=='noise':
            low=np.clip(low.astype(float)+np.random.default_rng(100+int(i)).normal(0,6,low.shape),0,255).astype(np.uint8)
            _,enc=cv2.imencode('.jpg',low,[cv2.IMWRITE_JPEG_QUALITY,60]);low=cv2.imdecode(enc,1)
        base=cv2.resize(low,(512,512),interpolation=cv2.INTER_LANCZOS4)
        old=pipe.wink_enhancer.apply_laplacian_pyramid_clarity(base,.35)
        new=pipe.process_image(low,preset_mode='Pure Quality',upscale=scale,clarity_strength=.5)
        assert not pipe._onnx_session_cache and not pipe._torch_model_cache
        assert new.shape==ref.shape
        delta=new.astype(int)-base.astype(int)
        np.testing.assert_array_equal(delta[:,:,0],delta[:,:,1]);np.testing.assert_array_equal(delta[:,:,1],delta[:,:,2])
        folder=out/f'{i}_{condition}';folder.mkdir()
        for label,image in [('reference',ref),('input',base),('old',old),('new',new)]:cv2.imwrite(str(folder/f'{label}.png'),image)
        row={'path':names[i],'source_sha256':hashlib.sha256((ROOT/'models/CodeFormer/datasets/ffhq/ffhq_512'/names[i]).read_bytes()).hexdigest(),'index':int(i),'condition':condition,'metrics':{}}
        for label,image in [('old',old),('new',new)]:
            row['metrics'][label]=eval.evaluate_pair(image,ref)
            row['metrics'][label]['change_mae']=float(np.abs(image.astype(float)-base).mean())
        report['samples'].append(row);save()
    print(f"Completed {names[i]} ({len(report['samples'])}/60)",flush=True)
 report['summary']={}
 for condition in ('soft','severe','noise','clean','motion'):
    rows=[r for r in report['samples'] if r['condition']==condition]
    values={}
    for metric in ('psnr','ssim','lpips','identity_similarity','change_mae'):
        paired=[r['metrics']['new'][metric]-r['metrics']['old'][metric] for r in rows if metric in r['metrics']['new'] and metric in r['metrics']['old']]
        values[metric]={'median_delta':float(np.median(paired)),'min_delta':min(paired),'coverage':len(paired)}
    values['lpips_relative_median_percent']=float(np.median([100*(r['metrics']['new']['lpips']/r['metrics']['old']['lpips']-1) for r in rows]))
    report['summary'][condition]=values
 report['status']='completed';save();print(json.dumps(report['summary'],indent=2),flush=True)
except Exception as e:
 report['status']='failed';report['error']=repr(e);save();raise
