# external 
import numpy as np

#internal 
import sys
import configparser
configparser.SafeConfigParser = configparser.ConfigParser
import time
from types import SimpleNamespace
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"  # deaktiviert Ausgabe komplett
# %% set parameters
sys.path.append(r'/g/prevedel/members/Goerlitz/projectsHPC/brillo')
import brilloFunctions_v06_cluster as bf
#mainPath = "/g/prevedel/members/Goerlitz/projectsHPC/brillo/results/"
mainPath = "/scratch/goerlitz/brilloCopy/768x3_00deg_"

s = time.time()


optExc = SimpleNamespace()
optExc.Nx = 768
optExc.Ny = optExc.Nx
optExc.Nz = optExc.Nx
optExc.dx = 0.05
optExc.dy = optExc.dx
optExc.dz = optExc.dx
optExc.NA = 0.2
optExc.n0 = 1.33
optExc.lam = 0.532

optDet = SimpleNamespace()
optDet.Nx = optExc.Nx
optDet.Ny = optDet.Nx
optDet.Nz = optDet.Nx
optDet.dx = optExc.dx
optDet.dy = optDet.dx
optDet.dz = optDet.dx
optDet.NA = 0.8
optDet.n0 = optExc.n0
optDet.lam = 0.580
optDet.angle = 00

# check if pixelsize smaller than Nyquist 
dxExc = optExc.lam/2/optExc.NA
dxDet = optDet.lam/2/optDet.NA
if np.min([dxDet, dxExc]) <= optExc.dx:
    print('..........................Warning ...................................... -> K-Space not Nyquist sampled for choosen NA and dx.')
#%% prepare propagation volume


# scatPath = 'C:\\Users\\Goerlitz\\Documents\\temp\\Tabea_mouseembryo_001_512.tif'
# scatVol1 = tiff.imread(scatPath)/10000
# scatVol = np.transpose(scatVol1, (1, 0, 2))
# sf = 0.200/optExc.dx
# scale_factors = (sf, sf, sf)
# scatVol = zoom(scatVol1, scale_factors, order=1)  # order=1 = linear interpolation

vol = bf.create_sphere_with_gaussian_noise(shape=(optExc.Nx, optExc.Nx, optExc.Nx),
                                        n_background=1.33,
                                        n_sphere=1.43,
                                        radius=64, #xxx256,
                                        noise_std=0.01)


padded_scatVol = bf.genPaddArray(optExc.Nx, optExc.Nx, optExc.Nx, vol)
sz, sy, sx = vol.shape
print("... propagation volume loaded/generated")
del vol

# %% pepare vols histo parameters and scatter propagator
psfE, psfD, theta, phi, bins, angles_rad, sexy, kxy = bf.prepPara(optExc, optDet)
print("... prepared PSFs")

#%% define steps

xsteps = 64
xrange = 608
xstepSize = round(xrange/(xsteps - 1))
xrange = 0 if xsteps == 1 else xrange
xstepSize = 0 if xsteps == 1 else round(xrange / (xsteps - 1))

ysteps = xsteps
yrange = xrange
yrange = 0 if ysteps == 1 else yrange
ystepSize = 0 if ysteps == 1 else round(yrange / (ysteps - 1))

zsteps = 1
zrange = xrange
zrange = 0 if zsteps == 1 else zrange
zstepSize = 0 if zsteps == 1 else round(zrange / (zsteps - 1))


#%% threading
comTheta = np.zeros((xsteps, ysteps, zsteps))
comPhi = np.zeros((xsteps, ysteps, zsteps))
stdTheta = np.zeros((xsteps, ysteps, zsteps))
stdPhi = np.zeros((xsteps, ysteps, zsteps))
meanTheta = np.zeros((xsteps, ysteps, zsteps))
meanPhi = np.zeros((xsteps, ysteps, zsteps))

# Use ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as executor:  # adjust workers to CPU cores
    futures = []
    bf.print_memory_usage(padded_scatVol=padded_scatVol, 
                          psfE=psfE, psfD=psfD, optExc=optExc, optDet=optDet,
                          mainPath=mainPath, theta=theta, phi=phi)
    for i in range(xsteps):
        start_x = round(sx - xrange/2) + i * xstepSize
        end_x   = start_x + sx
        for j in range(ysteps):
            start_y = round(sy - yrange/2) + j * ystepSize 
            end_y   = start_y + sy
            for w in range(zsteps):
                start_z = round(sz - zrange/2) + w * zstepSize
                end_z   = start_z + sz
                coo = [start_x, end_x, start_y, end_y, start_z, end_z]
                futures.append(executor.submit(bf.process_shift, coo, 
                                               padded_scatVol, psfE, psfD, 
                                               optExc, optDet, mainPath,
                                               theta, phi, i, j, w))
                
    total_tasks = len(futures)
    completed = 0
    for future in as_completed(futures):
        res = SimpleNamespace()
        res = future.result()
        comTheta[res.x, res.y, res.z] = res.thetaCOM
        comPhi[res.x, res.y, res.z] = res.phiCOM
        stdTheta[res.x, res.y, res.z] = res.thetaSTD
        stdPhi[res.x, res.y, res.z] = res.phiSTD
        meanTheta[res.x, res.y, res.z] = res.thetaMean
        meanPhi[res.x, res.y, res.z] = res.phiMean
        completed += 1
        print(f"[{completed}/{total_tasks}] Completed")
        
#%% save phi and theta
scatDim = optExc.dx

xRes = 1 / (scatDim / 10000)
yRes = 1 / (scatDim / 10000)

name = 'dTheta'
bf.saveDist(mainPath, 'comTheta', comTheta[:,:,0], xRes, yRes, scatDim)
bf.saveDist(mainPath, 'comPhi', comPhi[:,:,0], xRes, yRes, scatDim)
bf.saveDist(mainPath, 'stdTheta', stdTheta[:,:,0], xRes, yRes, scatDim)
bf.saveDist(mainPath, 'stdPhi', stdPhi[:,:,0], xRes, yRes, scatDim)
bf.saveDist(mainPath, 'meanTheta', meanTheta[:,:,0], xRes, yRes, scatDim)
bf.saveDist(mainPath, 'meanPhi', meanPhi[:,:,0], xRes, yRes, scatDim)