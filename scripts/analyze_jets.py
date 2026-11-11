"""Reads a result.vtk written by `peregrine run` and reports, per outlet, the mean exit velocity and angle,
then traces streamlines from the outlets and prints how their spread changes with depth below the duct
(use it to check whether jets converge where you expect).

usage: python3 -I scripts/analyze_jets.py out/result.vtk project.toml   (needs numpy)
"""
import sys, struct, re, tomllib
import numpy as np
def load(path):
    b = open(path,'rb').read()
    hdr = b[:600].decode('latin1')
    dims = tuple(int(x) for x in re.search(r'DIMENSIONS (\d+) (\d+) (\d+)', hdr).groups())
    origin = [float(x) for x in re.search(r'ORIGIN (\S+) (\S+) (\S+)', hdr).groups()]
    dx = float(re.search(r'SPACING (\S+)', hdr).group(1))
    nx,ny,nz = dims[0]-1, dims[1]-1, dims[2]-1
    n = nx*ny*nz
    pos = b.index(b'CELL_DATA'); pos = b.index(b'\n', pos)+1
    def scalar(pos):
        pos = b.index(b'\n', pos)+1   # SCALARS line
        pos = b.index(b'\n', pos)+1   # LOOKUP_TABLE
        a = np.frombuffer(b[pos:pos+4*n], dtype='>f4').astype(np.float32); return a, pos+4*n+1
    cell, pos = scalar(pos); speed, pos = scalar(pos); pres, pos = scalar(pos)
    pos = b.index(b'\n', pos)+1
    vel = np.frombuffer(b[pos:pos+12*n], dtype='>f4').astype(np.float32).reshape(-1,3)
    sh = (nz,ny,nx)
    return dict(cell=cell.reshape(sh), speed=speed.reshape(sh), pres=pres.reshape(sh), vel=vel.reshape(nz,ny,nx,3), origin=np.array(origin), dx=dx, dims=(nx,ny,nz))
def sample(d, p):
    g = (p - d['origin'])/d['dx'] - 0.5
    nx,ny,nz = d['dims']
    if g[0]<0 or g[1]<0 or g[2]<0 or g[0]>nx-1 or g[1]>ny-1 or g[2]>nz-1: return None
    i = np.floor(g).astype(int); t = g - i
    i1 = np.minimum(i+1, [nx-1,ny-1,nz-1])
    v = np.zeros(3)
    for dz in (0,1):
        for dy in (0,1):
            for dx_ in (0,1):
                w = (t[0] if dx_ else 1-t[0])*(t[1] if dy else 1-t[1])*(t[2] if dz else 1-t[2])
                v += w*d['vel'][(i1 if dz else i)[2], (i1 if dy else i)[1], (i1 if dx_ else i)[0]]
    return v
def trace(d, p, steps=400, ds=0.5):
    pts=[p.copy()]
    for _ in range(steps):
        v = sample(d,p)
        if v is None or np.linalg.norm(v)<0.05: break
        k1 = v/np.linalg.norm(v)
        def dr(q):
            w = sample(d,q); 
            return None if w is None or np.linalg.norm(w)<0.05 else w/np.linalg.norm(w)
        k2 = dr(p+0.5*ds*k1); 
        if k2 is None: break
        k3 = dr(p+0.5*ds*k2)
        if k3 is None: break
        k4 = dr(p+ds*k3)
        if k4 is None: break
        p = p + ds*(k1+2*k2+2*k3+k4)/6; pts.append(p.copy())
    return np.array(pts)
if __name__=='__main__':
    vtk, proj = sys.argv[1], sys.argv[2]
    d = load(vtk)
    cfg = tomllib.load(open(proj,'rb'))
    ports = cfg['ports']
    outs = [i for i,p in enumerate(ports) if 'PressureOutlet' in p['kind']]
    print('grid', d['dims'], 'origin', d['origin'])
    # outlet exit direction: mean velocity of the fluid cell layer just outside each outlet plane (below, -z)
    allseeds=[]
    for i in outs:
        code = 2+i
        zz,yy,xx = np.where(d['cell']==code)
        if len(zz)==0: continue
        v = d['vel'][zz,yy,xx].mean(axis=0)
        sp = np.linalg.norm(v)
        cen = d['origin'] + (np.array([xx.mean(),yy.mean(),zz.mean()])+0.5)*d['dx']
        ang = np.degrees(np.arctan2(np.hypot(v[0],v[1]), -v[2]))
        print(f"{ports[i]['name']:6s} cells {len(zz):4d} centre x={cen[0]:7.1f} y={cen[1]:8.1f} z={cen[2]:6.1f}  mean exit vel ({v[0]:5.2f},{v[1]:5.2f},{v[2]:5.2f}) |v|={sp:4.2f} m/s, {ang:4.1f} deg off straight-down, x-drift {v[0]/max(-v[2],1e-6):+.2f}")
        for (z,y,x) in list(zip(zz,yy,xx))[::max(1,len(zz)//6)]:
            allseeds.append((i, d['origin'] + (np.array([x,y,z])+0.5)*d['dx'] - np.array([0,0,0.0])))
    # trace each seed down; record x of lines at depths below the duct bottom plane
    zb = d['origin'][2] + (min(np.where(d['cell']==2+outs[0])[0]))*d['dx'] if outs else 0
    print('duct bottom plane z ~', round(float(zb),1))
    res={}
    for i,p in allseeds:
        tr = trace(d, p)
        for depth in (0,6,12.7,20):
            idx = np.where(tr[:,2] <= zb - depth)[0]
            if len(idx): res.setdefault(depth,[]).append((tr[idx[0],0], tr[idx[0],1]))
    for depth,v in sorted(res.items()):
        a=np.array(v); print(f"depth {depth:5.1f} mm: {len(a):3d} lines  x spread {a[:,0].min():7.1f}..{a[:,0].max():7.1f} (std {a[:,0].std():5.1f})  y spread {a[:,1].min():8.1f}..{a[:,1].max():8.1f} (std {a[:,1].std():5.1f})")
