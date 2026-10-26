"""Compare saved constructor metrics; numerical execution only on KEVIN."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);args=p.parse_args()
    checks={};passed=True
    with np.load(args.baseline) as a,np.load(args.candidate) as b:
        assert a.files==b.files
        for key in a.files:
            av,bv=np.asarray(a[key],np.float64),np.asarray(b[key],np.float64)
            error=float(np.max(np.abs(av-bv))/max(1.,np.max(np.abs(av))))
            assert np.isfinite(av).all() and np.isfinite(bv).all()
            check={'relative_max_error':error,'bitidentical':bool(np.array_equal(av,bv)),'passed':bool(error<2e-5)}
            if key=='A':
                eigen=np.linalg.eigvalsh(bv);check['eigen_min']=float(eigen.min());check['eigen_max']=float(eigen.max())
                check['passed'] &=bool(eigen.min()>=np.exp(-4)*(1-1e-3) and eigen.max()<=np.exp(4)*(1+1e-3))
            elif key=='B':check['passed'] &=bool(np.max(np.abs(bv))<=4*(1+1e-3))
            passed &= check['passed'];checks[key]=check
    result={'arguments':vars(args),'checks':checks,'passed':passed,
        'input_sha256':{name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('baseline',args.baseline),('candidate',args.candidate)]},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert passed


if __name__=='__main__':main()
