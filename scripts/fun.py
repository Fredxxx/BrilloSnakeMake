import os
import time
import sys
from types import SimpleNamespace
import json
import argparse
import logging
import gc
import tifffile as tiff
import numpy as np
from numpy.fft import fftn, fftshift, fftfreq
from scipy.ndimage import zoom, center_of_mass
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
import configparser
configparser.SafeConfigParser = configparser.ConfigParser
import biobeam as bb
import matplotlib.pyplot as plt

#%% helper functions --- not directly called by snakemake

def genPaddArray(s, vol, pad):
    
    Z, Y, X = s * pad, s * pad, s
    #padded_scatVol = np.random.normal(1.33335, 0.00074, size=(Z, Y, X)).astype(np.float16)
    padded_scatVol = np.random.normal(1.3375, 0.0025, size=(Z, Y, X)).astype(np.float32)
    
    vz, vy, vx = vol.shape
    
    
    def get_indices(target_size, source_size):
        start_t = max(0, (target_size - source_size) // 2)
        start_s = max(0, (source_size - target_size) // 2)
        length = min(target_size, source_size)
        return start_t, start_s, length

    zt, zs, zl = get_indices(Z, vz)
    yt, ys, yl = get_indices(Y, vy)
    xt, xs, xl = get_indices(X, vx)

    padded_scatVol[zt:zt+zl, yt:yt+yl, xt:xt+xl] = vol[zs:zs+zl, ys:ys+yl, xs:xs+xl]
    
    return padded_scatVol

def plot_max_projections(volume, voxel_size=(1.0, 1.0, 1.0), cmap='hot',
                         title="Max Intensity Projections", space="real",
                         globalPlotScaling=None):
    """
    Plottet Maximalprojektionen eines 3D-Volumes im Ortsraum (real) oder Frequenzraum (fft),
    mit einheitlicher Farbskala (automatisch oder durch globalPlotScaling gesetzt) und Colorbar.
    
    globalPlotScaling: [v1, vMinReal, vMaxReal, vMinImag, vMaxImag]
        v1=1 -> Werte übernehmen, 0 -> automatisch berechnen
        space="real" -> vMinReal/vMaxReal verwenden
        space="fft" -> vMinImag/vMaxImag verwenden
    """
    dz, dy, dx = voxel_size
    Z, Y, X = volume.shape

    # Extents definieren
    if space == "fft":
        unit = "1/µm"
        dfz = 1.0 / (Z * dz)
        dfy = 1.0 / (Y * dy)
        dfx = 1.0 / (X * dx)
        extent_xy = [-X//2*dfx, X//2*dfx, -Y//2*dfy, Y//2*dfy]
        extent_xz = [-X//2*dfx, X//2*dfx, -Z//2*dfz, Z//2*dfz]
        extent_yz = [-Y//2*dfy, Y//2*dfy, -Z//2*dfz, Z//2*dfz]
    else:
        unit = "µm"
        extent_xy = [-X//2*dx, X//2*dx, -Y//2*dy, Y//2*dy]
        extent_xz = [-X//2*dx, X//2*dx, -Z//2*dz, Z//2*dz]
        extent_yz = [-Y//2*dy, Y//2*dy, -Z//2*dz, Z//2*dz]

    # Max-Projektionen
    max_xy = np.max(volume, axis=0)
    max_xz = np.max(volume, axis=1)
    max_yz = np.max(volume, axis=2)

    # Farbskala bestimmen
    if globalPlotScaling and globalPlotScaling[0] == 1:
        if space == "real":
            vmin, vmax = globalPlotScaling[1], globalPlotScaling[2]
        else:
            vmin, vmax = globalPlotScaling[3], globalPlotScaling[4]
    else:
        # Automatisch aus Daten
        vmin = min(max_xy.min(), max_xz.min(), max_yz.min())
        vmax = max(max_xy.max(), max_xz.max(), max_yz.max())

    # Plot erstellen
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title} ({space.upper()} space)", fontsize=16)

    im0 = axes[0].imshow(max_xy, cmap=cmap, extent=extent_xy, origin='lower',
                         aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title('Z-Projection (XY)')
    axes[0].set_xlabel(f'X ({unit})')
    axes[0].set_ylabel(f'Y ({unit})')

    im1 = axes[1].imshow(max_xz, cmap=cmap, extent=extent_xz, origin='lower',
                         aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('Y-Projection (XZ)')
    axes[1].set_xlabel(f'X ({unit})')
    axes[1].set_ylabel(f'Z ({unit})')

    im2 = axes[2].imshow(max_yz, cmap=cmap, extent=extent_yz, origin='lower',
                         aspect='auto', vmin=vmin, vmax=vmax)
    axes[2].set_title('X-Projection (YZ)')
    axes[2].set_xlabel(f'Y ({unit})')
    axes[2].set_ylabel(f'Z ({unit})')

    # Colorbar rechts neben das rechte Bild
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    return fig, axes
    
def add_command(subParsers, name, func, parents=None):
    subparser = subParsers.add_parser(name, parents=parents or [])
    subparser.set_defaults(func=func)
    return subparser

def fftcpuPS(psf):
    return np.abs(fftshift(fftn(psf)))**2

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fitGauss(cent, proj):    
    prop, pcov = curve_fit(gauss, cent, proj, p0=[proj.max(), cent[np.argmax(proj)], 5])
    return prop, pcov

def save_theta_phi_plot(hist, res, h, path, a):
    
    _, theMu, theSig = res.theFit
    _, phiMu, phiSig = res.phiFit
    
    fig = plt.figure(figsize=(8, 8))

    ax_main  = fig.add_axes([0.1, 0.1, 0.6, 0.6])
    ax_top   = fig.add_axes([0.1, 0.72, 0.6, 0.18], sharex=ax_main)
    ax_right = fig.add_axes([0.72, 0.1, 0.18, 0.6], sharey=ax_main)

    # Main image
    im = ax_main.imshow(
        hist,
        origin='lower',
        extent=[
            res.thetaBins[0] + a,
            res.thetaBins[-1] + a,
            res.phiBins[0] - a,
            res.phiBins[-1] - a
        ],
        aspect='auto',
        cmap='viridis'
    )
    ax_main.set_xlabel("Theta")
    ax_main.set_ylabel("Phi")

    # Top plot (theta projection)
    ax_top.plot(h.theC + a, h.theProj, label="Proj Theta")
    ax_top.plot(h.theC + a, gauss(h.theC, *res.theFit), 'r--', label="Gauss Fit")
    ax_top.set_ylabel("Σ over Phi")
    ax_top.legend()
    ax_top.tick_params(labelbottom=False)
    ax_top.text(0.05, 0.7, f"μ={theMu:.2f}, σ={theSig:.2f}", transform=ax_top.transAxes)

    # Right plot (phi projection)
    ax_right.plot(h.phiProj - a, h.phiC - a, label="Proj Phi")
    ax_right.plot(gauss(h.phiC, *res.phiFit), h.phiC - a, 'r--', label="Gauss Fit")
    ax_right.set_xlabel("Σ over Theta")
    ax_right.legend()
    ax_right.tick_params(labelleft=False)
    ax_right.text(0.05, 0.7, f"μ={phiMu:.2f}, σ={phiSig:.2f}", transform=ax_right.transAxes)

    # Save instead of show
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


#%% active functions -- which are called by snakemake

def loadPara(args):
    inPath = args.input[0]
    outPath = args.output[0]
    
    optExc = SimpleNamespace()
    optDet = SimpleNamespace()
    optGen = SimpleNamespace()
    adv = SimpleNamespace()
    scanPara = SimpleNamespace()
    context = {
        "optExc": optExc,
        "optDet": optDet,
        "optGen": optGen,
        "adv": adv,
        "scanPara": scanPara
    }
    with open(inPath, 'r') as f:
        dat = f.read()
    
    exec(dat, globals(), context)
    
    result = {
        "mainPath": context.get("mainPath"),
        "name": context.get("name"),
        "scatPath": context.get("scatPath"),
        "mode": context.get("mode"),
        "optExc": vars(optExc), 
        "optDet": vars(optDet),
        "optGen": vars(optGen),
        "adv": vars(adv),
        "scanPara": vars(scanPara),
    }
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    with open(outPath, 'w') as f:
        json.dump(result, f, indent=4)
    print("----- para loaded -----", flush=True)

def loadPadSampleVol(args):
    inPath = args.input[0]
    outPath = args.output[0]
    
    # load image, swap axis, zoom and pad
    with open(inPath, 'r') as f:
        js = json.load(f)
    #print(js)
    scatVol = tiff.imread(js["scatPath"])/10000 + js["adv"]["nOff"]
    scatVol = np.swapaxes(scatVol, 0, 2)
    #sf = 0.23/js["optExc"]["d"]
    sf = 0.23*4//js["optExc"]["d"]
    scale_factors = (sf, sf, 4*sf)
    scatVol = zoom(scatVol, scale_factors, order=1)  # order=1 = linear interpolation
    
    if js["adv"]["showImg"] == 2:
        plot_max_projections(scatVol, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="scatVol (loadPadSampleVol)", space="real")
        
    scatVol = genPaddArray(js["optExc"]["N"], scatVol, js["adv"]["pad"]) 
    
    if js["adv"]["showImg"] in [1, 2]:
        plot_max_projections(scatVol, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="scatVol (loadPadSampleVol)", space="real")
    
    tiff.imwrite(outPath, scatVol)
    print("----- sample loaded -----", flush=True)

def genExcPSF(args):
    inPath = args.input[0]
    outPreal = args.output[0]
    outPimag = args.output[1]
    
    with open(inPath, 'r') as f:
        js = json.load(f)
    
    if js["mode"] == "GaussBeam":
        params = {
            "shape": (js["optExc"]["N"],) * 3,
            "units": (js["optExc"]["d"],) * 3,
            "lam": js["optExc"]["lam"],
            "NA": js["optExc"]["NA"],
            "n0": js["optExc"]["n0"],
            "return_all_fields": True,
            "n_integration_steps": 100
        }
        _ , hE, _, _ = bb.focus_field_beam(**params)
    elif js["mode"] == "BesselBeam":
        params = {
            "shape": (js["optExc"]["N"],) * 3,
            "units": (js["optExc"]["d"],) * 3,
            "lam": js["optExc"]["lam"],
            "NA": [js["optExc"]["besselNAin"], js["optExc"]["besselNAout"]],
            "n0": js["optExc"]["n0"],
            "return_all_fields": True,
            "n_integration_steps": 100
        }
        _ ,hE, _, _ = bb.focus_field_beam(**params)
    elif js["mode"] == "GaussSheet":
        params = {
            "shape": (js["optExc"]["N"],) * 3,
            "units": (js["optExc"]["d"],) * 3,
            "lam": js["optExc"]["lam"],
            "NA": js["optExc"]["NA"],
            "n0": js["optExc"]["n0"],
            "return_all_fields": True,
            "n_integration_steps": 100
        }
        _ ,hE, _, _ = bb.focus_field_cylindrical(**params)
    else:
        hE = np.zeros((js["optExc"]["N"],) * 3)
    
    if js["adv"]["showImg"] == 1:
        plot_max_projections(np.abs(hE)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf excitation intensity (genExcPSF)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])
    elif js["adv"]["showImg"] == 2:
        plot_max_projections(np.abs(hE)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf excitation intensity (genExcPSF)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])
        plot_max_projections(np.angle(hE), voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf excitation phase (genExcPSF)", space="ftt", globalPlotScaling=js["adv"]["globalPlotScaling"]) 
        
    # hE1 = hE.real.astype(np.float32) + 1j*hE.imag.astype(np.float32)
    
    # if js["adv"]["showImg"] == 1:
    #     plot_max_projections(np.abs(hE1)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf excitation intensity (genExcPSF)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])
    # elif js["adv"]["showImg"] == 2:
    #     plot_max_projections(np.abs(hE1)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf excitation intensity (genExcPSF)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])
    #     plot_max_projections(np.angle(hE1), voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf excitation phase (genExcPSF)", space="ftt", globalPlotScaling=js["adv"]["globalPlotScaling"]) 
        
    tiff.imwrite(outPreal, hE.real.astype(np.float32))
    tiff.imwrite(outPimag, hE.imag.astype(np.float32))
    print("----- excitation PSF generated -----", flush=True) 

def genDetPSF(args):  
    inPath = args.input[0]
    outPreal = args.output[0]
    outPimag = args.output[1]
    
    with open(inPath, 'r') as f:
        js = json.load(f)
        
    params = {
        "shape": (js["optExc"]["N"],) * 3,
        "units": (js["optExc"]["d"],) * 3,
        "lam": js["optDet"]["lam"],
        "NA": js["optDet"]["NA"],
        "n0": js["optExc"]["n0"],
        "return_all_fields": True,
        "n_integration_steps": 100
    }
    _ ,hD, _, _ = bb.focus_field_beam(**params)
    
    if js["adv"]["showImg"] == 1:
        plot_max_projections(np.abs(hD)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf detection intensity (genDetPSF)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])
    elif js["adv"]["showImg"] == 2:
        plot_max_projections(np.abs(hD)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf detection intensity (genDetPSF)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])
        plot_max_projections(np.angle(hD), voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="ps detection phase (genDetPSF)", space="ftt", globalPlotScaling=js["adv"]["globalPlotScaling"])   

    tiff.imwrite(outPreal, hD.real.astype(np.float32))
    tiff.imwrite(outPimag, hD.imag.astype(np.float32))
    print("----- detection PSF generated -----", flush=True)

def genAngleSpace(args):
    # xxx --- could be further devided into phi an theta generation 
    inPath = args.input[0]
    outPtheta = args.output[0]
    outPphi = args.output[1]
    
    with open(inPath, 'r') as f:
        js = json.load(f)
    
    # define k space
    kx = fftshift(fftfreq(js["optExc"]["N"], d=js["optExc"]["d"])) * 2 * np.pi
    KZ, KY, KX = np.meshgrid(kx, kx, kx, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2) + 1e-12  # avoid divide by 0
    
    #Theta - interesting angle - polar angle
    theta = np.rad2deg(np.arccos(KZ / k_mag))  # polar angle
    if js["adv"]["showImg"] in [1, 2]:
        plot_max_projections(theta, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="theta vol (genAngleSpace)", space="real")
    tiff.imwrite(outPtheta, theta.astype(np.float32))
    del theta, KZ, k_mag
    gc.collect()
    
    # Phi - less interesting (radial symmetric), azimutal
    phi = np.rad2deg(np.arctan2(KY, KX))       # azimuthal angle
    if js["adv"]["showImg"] == 2:
        plot_max_projections(phi, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="theta vol (genAngleSpace)", space="real")
    tiff.imwrite(outPphi, phi.imag.astype(np.float32))
    
    print("----- angle space generated -----", flush=True)
    
def genIDXs(args):
    inPath = args.input[0]
    outPath = args.output[0]
    
    with open(inPath, 'r') as f:
        js = json.load(f)

    sPad = js["optExc"]["N"]*js["adv"]["pad"]
    sVol = js["optExc"]["N"]
    
    xsteps = js["scanPara"]["xSteps"]
    xrange = js["scanPara"]["xRange"]
    ysteps = xsteps 
    yrange = xrange
    zsteps = 1
    zrange = xrange

    # calculate step size
    xstepSize = 0 if xsteps == 1 else round(xrange / (xsteps - 1))
    ystepSize = 0 if ysteps == 1 else round(yrange / (ysteps - 1))
    zstepSize = 0 if zsteps == 1 else round(zrange / (zsteps - 1))

    # set scan coordinates
    coordinate_sets = []
    for i in range(xsteps):
        start_x = round(sPad/2 - (sVol+xrange)/2) + i * xstepSize
        end_x = start_x + sVol
        for j in range(ysteps):
            start_y = round(sPad/2 - (sVol+yrange)/2) + j * ystepSize
            end_y = start_y + sVol
            for w in range(zsteps):
                start_z = 0 # xxx not sure but works
                end_z = start_z + sVol
                
                coo = [start_y, end_y, start_x, end_x, start_z, end_z, ]
                coordinate_sets.append(coo)

    # constitute scanParas and save as json
    scanData = {
        "idxMax": len(coordinate_sets),
        "idxVector": list(range(len(coordinate_sets))),
        "x": {"steps": xsteps, "range": xrange, "stepSize": xstepSize},
        "y": {"steps": ysteps, "range": yrange, "stepSize": ystepSize},
        "z": {"steps": zsteps, "range": zrange, "stepSize": zstepSize},
        "coo": coordinate_sets
    }
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    with open(outPath, 'w') as f:
        json.dump(scanData, f, indent=4)
    print("----- scan para prepared -----", flush=True)

def propExcVol(args):
    # def args
    inPath = args.input[0]
    inScanPar = args.input[1]
    inPSFreal = args.input[2]
    inPSFimag = args.input[3]
    inPropVol = args.input[4]
    outPSFreal = args.output[0]
    outPSFimag = args.output[1]
    
    # load general + scan parameters
    with open(inPath, 'r') as f:
        js = json.load(f)
    with open(inScanPar, 'r') as f:
        sp = json.load(f)
    i = int(outPSFreal.split("_")[-1].split(".")[0]) # identify idx    
        
    # load volumes and make complex field    
    psfReal = tiff.imread(inPSFreal)
    psfImag = tiff.imread(inPSFimag)
    proVol = tiff.imread(inPropVol)
    psf = psfReal + 1j*psfImag 
    
    # propaget though medium
    t = proVol[sp["coo"][i][0]:sp["coo"][i][1], sp["coo"][i][2]:sp["coo"][i][3], sp["coo"][i][4]:sp["coo"][i][5]]
    os.makedirs("results/deb", exist_ok=True)
    tiff.imwrite(f"results/deb/propExc_t_{i}.tif", t) # xxx deb
    del proVol; gc.collect()
    t = bb.Bpm3d(dn=t, units = (js["optExc"]["d"],)*3, lam=js["optExc"]["lam"]/js["optExc"]["n0"])
    psfScat = t.propagate(u0 = psf[0,:,:]) 
    del psf, t; gc.collect()
    
    # rotate
    if js["optDet"]["angle"] == 0:
        psfScat = np.rot90(psfScat, k=1, axes=(1, 2))
    elif js["optDet"]["angle"] != 90:
        print("----- error propExcVol: optDet.angle only supported for 0 and 90. Continue with 0 case. -----", flush=True)
    tiff.imwrite(f"results/deb/propExc_psfScat_{i}.tif", np.abs(psfScat)**2) # xxx deb
    
    # plotting
    if js["adv"]["showImg"] == 1:
        plot_max_projections(np.abs(psfScat)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf excitation intensity (propExcVol)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])
    elif js["adv"]["showImg"] == 2:
        plot_max_projections(np.abs(psfScat)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf excitation intensity (propExcVol)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])
        plot_max_projections(np.angle(psfScat), voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf excitation phase (propExcVol)", space="ftt", globalPlotScaling=js["adv"]["globalPlotScaling"]) 
    
    # save
    tiff.imwrite(outPSFreal, psfScat.real.astype(np.float32))
    tiff.imwrite(outPSFimag, psfScat.imag.astype(np.float32)) 
    print(f"----- excitation propagated idx={i} -----", flush=True)
        
def propDetVol(args):
    # def args
    inPath = args.input[0]
    inScanPar = args.input[1]
    inPSFreal = args.input[2]
    inPSFimag = args.input[3]
    inPropVol = args.input[4]
    outPSFreal = args.output[0]
    outPSFimag = args.output[1]
    
    # load general + scan parameters
    with open(inPath, 'r') as f:
        js = json.load(f)
    with open(inScanPar, 'r') as f:
        sp = json.load(f)
    i = int(outPSFreal.split("_")[-1].split(".")[0])

    # load volumes and make complex field     
    psfReal = tiff.imread(inPSFreal)
    psfImag = tiff.imread(inPSFimag)
    proVol = tiff.imread(inPropVol)
    psf = psfReal + 1j*psfImag
    
    # propagate thorugh medium and rot
    if js["optDet"]["angle"] == 0:
        t = proVol[sp["coo"][i][0]:sp["coo"][i][1], sp["coo"][i][2]:sp["coo"][i][3], sp["coo"][i][4]:sp["coo"][i][5]]
        del proVol; gc.collect()
        tiff.imwrite(f"results/deb/propDet_t_{i}.tif", t) # xxx deb
        t = bb.Bpm3d(dn=t, units = (js["optExc"]["d"],)*3, lam=js["optDet"]["lam"]/js["optExc"]["n0"])
        psfScat = t.propagate(u0 = psf[0,:,:])
        del psf, t; gc.collect()    
        psfScat = np.rot90(psfScat, k=1, axes=(1, 2))
    elif js["optDet"]["angle"] == 90:
        t = proVol[sp["coo"][i][0]:sp["coo"][i][1], sp["coo"][i][2]:sp["coo"][i][3], sp["coo"][i][4]:sp["coo"][i][5]]
        t = np.rot90(t, k=1, axes=(1, 2))
        del proVol; gc.collect()
        tiff.imwrite(f"results/deb/propDet_t_{i}.tif", t) # xxx deb
        t = bb.Bpm3d(dn=t, units = (js["optExc"]["d"],)*3, lam=js["optDet"]["lam"]/js["optExc"]["n0"])
        psfScat = t.propagate(u0 = psf[0,:,:])
        del psf, t; gc.collect() 
        psfScat = np.rot90(psfScat, k=3, axes=(1, 2)) # rotates 90 back (270°)
    else:
        print("----- error propExcVol: optDet.angle only supported for 0 and 90. Continue with 0 case. -----", flush=True)
    tiff.imwrite(f"results/deb/propDet_psfScat_{i}.tif", np.abs(psfScat)**2) # xxx deb
    
    # plotting
    if js["adv"]["showImg"] == 1:
        plot_max_projections(np.abs(psfScat)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf detection intensity (propExcVol)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])
    elif js["adv"]["showImg"] == 2:
        plot_max_projections(np.abs(psfScat)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf detection intensity (propExcVol)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])
        plot_max_projections(np.angle(psfScat), voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf detection phase (propExcVol)", space="ftt", globalPlotScaling=js["adv"]["globalPlotScaling"]) 
    
    # save 
    tiff.imwrite(outPSFreal, psfScat.real.astype(np.float32))
    tiff.imwrite(outPSFimag, psfScat.imag.astype(np.float32)) 
    print(f"----- detection propagated idx={i} -----", flush=True)
        
def genSysPSF(args):
    # def args
    inPath = args.input[0]
    inPSFrealExc = args.input[1]
    inPSFimagExc = args.input[2]
    inPSFrealDet = args.input[3]
    inPSFimagDet = args.input[4]
    outPSFreal = args.output[0]
    outPSFimag = args.output[1]
    
    # load general parameters
    with open(inPath, 'r') as f:
        js = json.load(f)
    i = int(outPSFreal.split("_")[-1].split(".")[0])
    
    # load volumes and make complex field (exciation)  
    psfEreal = tiff.imread(inPSFrealExc)
    psfEimag = tiff.imread(inPSFimagExc)
    psfE = psfEreal + 1j*psfEimag
    del psfEreal, psfEimag; gc.collect()
    
    # load volumes and make complex field (detection)  
    psfDreal = tiff.imread(inPSFrealDet)
    psfDimag = tiff.imread(inPSFimagDet)
    psfD = psfDreal + 1j*psfDimag
    del psfDreal, psfDimag; gc.collect()
    
    # gen system PSF 
    psfSys = psfE * psfD
    
    # plotting
    if js["adv"]["showImg"] == 1:
        plot_max_projections(np.abs(psfSys)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf system intensity (genPS)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])    
    
    # save 
    tiff.imwrite(outPSFreal, psfSys.real.astype(np.float32))
    tiff.imwrite(outPSFimag, psfSys.imag.astype(np.float32))
    print(f"----- psf sys generated idx={i} -----", flush=True)

def genSysPS(args):
    # def args
    inPath = args.input[0]
    inPSFreal = args.input[1]
    inPSFimag = args.input[2]
    outPS = args.output[0]
    
    # load general parameters
    with open(inPath, 'r') as f:
        js = json.load(f)
    i = int(outPS.split("_")[-1].split(".")[0])
    
    # load volume
    psfReal = tiff.imread(inPSFreal)
    psfImag = tiff.imread(inPSFimag)
    psf = psfReal + 1j*psfImag
    del psfReal, psfImag; gc.collect()
    
    # calc power spectrum
    ps = fftcpuPS(psf)
    tiff.imwrite(f"results/deb/genSysPS_ps_{i}.tif", ps) # xxx deb
    
    # plotting
    if js["adv"]["showImg"] == 1:
        plot_max_projections(np.abs(ps)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="power spectrum (genPS)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])    
    
    # save
    tiff.imwrite(outPS, ps)
    print(f"----- ps sys generated idx={i} -----", flush=True)
        
def genHisto(args):
    # def args
    inPath = args.input[0]
    
    # load general parameters
    with open(inPath, 'r') as f:
        js = json.load(f)
    
    if js["calc"]["sys"] == 1:
        genHistoCalc(args, 0)
    elif js["calc"]["exc"] == 1:
        genHistoCalc(args, 1)
    elif js["calc"]["det"] == 1:
        genHistoCalc(args, 2)

def genHistoSys(args):
    inPath = args.input[0]
    if js["calc"]["sys"] == 1:
        inSysPS = args.input[3]
        ps = tiff.imread(inPS)  
        genHistoCalc(args, ps)
    
def genHistoCalc(args, ps):        
    # def args
    inPath = args.input[0]
    inThetaVol = args.input[1]
    inPhiVol = args.input[2]
    inSysPS = args.input[3]
    inExcReal = args.input[4]
    inExcImag = args.input[5]  
    inExcReal = args.input[4]
    inExcImag = args.input[5] 
    outResAngle = args.output[0]
    
    # load general parameters
    with open(inPath, 'r') as f:
        js = json.load(f)
    i = int(outResAngle.split("_")[-1].split(".")[0])
    res = SimpleNamespace()
    res.scatAng = SimpleNamespace()
    h = SimpleNamespace()
    
    # load volume and bins
    theta = tiff.imread(inThetaVol)
    phi = tiff.imread(inPhiVol)
    if mod == 0:
        ps = tiff.imread(inPS)
    elif mod ==1:
        ps = 
    theBins = np.linspace(*js["calc"]["theBins"])
    phiBins = np.linspace(*js["calc"]["phiBins"])
    
    # volume to vector
    thetaFlat = theta.ravel()
    phiFlat = phi.ravel()
    psFlat = ps.ravel()

    # weighted histogram
    hist_sum, _, _ = np.histogram2d(
        thetaFlat, phiFlat, bins=[theBins, phiBins], weights=psFlat)

    # counts per pixel
    counts, _, _ = np.histogram2d(
        thetaFlat, phiFlat, bins=[theBins, phiBins])

    # pixel power normalised to pixel counts
    with np.errstate(divide='ignore', invalid='ignore'):
        res.hist = hist_sum / counts
        res.hist[counts == 0] = 0
    
    # calc space center of mass
    com = tuple(np.round(center_of_mass(ps)).astype(int))
    res.scatAng.comPix = com
    res.scatAng.comTheta = float(theta[com])
    res.scatAng.comPhi = float(phi[com])
    
    #calc space variance
    coords = np.indices(ps.shape).reshape(3, -1)
    with np.errstate(divide='ignore', invalid='ignore'):
        res.scatAng.varTheta = np.average((thetaFlat - theta[com])**2, weights=psFlat)
        res.scatAng.varPhi   = np.average((phiFlat   - phi[com])**2,   weights=psFlat)
        res.scatAng.varPixXYZ = np.average((coords.T - com)**2, axis=0, weights=psFlat)
        res.scatAng.varPix = np.sqrt(np.sum(res.scatAng.varPixXYZ))

    h.theC = 0.5 * (res.thetaBins[:-1] + res.thetaBins[1:])
    h.phiC   = 0.5 * (res.phiBins[:-1] + res.phiBins[1:]) 
    h.theProj = res.hist.T.sum(axis=0)
    h.phiProj = res.hist.T.sum(axis=1)

    res.theFit, _ = fitGauss(h.theC, h.theProj)
    res.phiFit, _ = fitGauss(h.phiC, h.phiProj)
    
    # save
    if js["calc"]["fig"] == 1:
        path = f"../results/hist_{i}.tif"
        save_theta_phi_plot(res, h, path, js["optDet"]["angle"])
    os.makedirs(os.path.dirname(outResAngle), exist_ok=True)
    with open(outResAngle, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"----- angle histo generated idx={i} -----", flush=True)
    
def angles2Brillo(args):
    # inJSON
    # inResHisto_{idx} 
    # outResHistoBS_{idx} (theta. and phi. to BS)  -> not temp 
    print("angles2Brillo")

def constImag(args):
    # inJSON
    # inScanPar
    # outResHistoBS_{idx} 
    # outImag (mullit dimensional)
    print("constImag")

    
#%% main   
if __name__ == "__main__":
    
    # io parser
    ioParser = argparse.ArgumentParser(add_help=False)
    ioParser.add_argument("--input", nargs="+", required=True)
    ioParser.add_argument("--output", nargs="+", required=True)

    # function parser
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda args: parser.print_help())
    
    # add subparser to finction parser and commands -> function
    subParsers = parser.add_subparsers(required=True)
    add_command(subParsers, "loadPara", loadPara, parents=[ioParser])
    add_command(subParsers, "loadPadSampleVol", loadPadSampleVol, parents=[ioParser])
    add_command(subParsers, "genExcPSF", genExcPSF, parents=[ioParser])
    add_command(subParsers, "genDetPSF", genDetPSF, parents=[ioParser])
    add_command(subParsers, "genAngleSpace", genAngleSpace, parents=[ioParser])
    add_command(subParsers, "genIDXs", genIDXs, parents=[ioParser])
    add_command(subParsers, "propExcVol", propExcVol, parents=[ioParser])
    add_command(subParsers, "propDetVol", propDetVol, parents=[ioParser])
    add_command(subParsers, "genSysPSF", genSysPSF, parents=[ioParser])
    add_command(subParsers, "genSysPS", genSysPS, parents=[ioParser])
    add_command(subParsers, "genHisto", genHisto, parents=[ioParser])
    
    if len(sys.argv) <= 1:
        print("---- debug mode ----")
        
        # list of paths
        p = SimpleNamespace(
                paraTXT = "../data/para.txt",
                paraJSON = "../results/01_paraTemp.json",
                propVol  = "../results/02_propVol.tif",
                hEreal = "../results/02_psfEreal.tif",
                hEimag = "../results/02_psfEimag.tif",
                hDreal = "../results/02_psfDreal.tif",
                hDimag = "../results/02_psfDimag.tif",
                thetaVol = "../results/02_thetaVol.tif",
                phiVol   = "../results/02_phiVol.tif",
                scanPara = "../results/02_scanPara.json",
                hErealScat = "../results/03_I_psfErealScat_0.tif",
                hEimagScat = "../results/03_I_psfEimagScat_0.tif",
                hDrealScat = "../results/03_I_psfDrealScat_0.tif",
                hDimagScat = "../results/03_I_psfDimagScat_0.tif",
                psfSysReal = "../results/03_II_psfSysReal_0.tif",
                psfSysImag = "../results/03_II_psfSysImag_0.tif",
                psSys = "../results/03_II_psSys_0.tif",
                resAngles = "../results/03_III_resAngles_0.tif"
             )
             
        # define in and out
        dc1 = ["loadPara", "--input", p.paraTXT, "--output", p.paraJSON]
        dc2 = ["loadPadSampleVol", "--input", p.paraJSON, "--output", p.propVol]
        dc3 = ["genExcPSF", "--input", p.paraJSON, "--output", p.hEreal, p.hEimag]
        dc4 = ["genDetPSF", "--input", p.paraJSON, "--output", p.hDreal, p.hDimag]
        dc5 = ["genAngleSpace", "--input", p.paraJSON, "--output", p.thetaVol, p.phiVol]
        dc6 = ["genIDXs", "--input", p.paraJSON, "--output", p.scanPara]
        dc7 = ["propExcVol", "--input", p.paraJSON, p.scanPara, p.hEreal, p.hEimag, p.propVol, "--output", p.hErealScat, p.hEimagScat]
        dc8 = ["propDetVol", "--input", p.paraJSON, p.scanPara, p.hDreal, p.hDimag, p.propVol, "--output", p.hDrealScat, p.hDimagScat]
        dc9 = ["genSysPSF", "--input", p.paraJSON, p.hErealScat, p.hEimagScat, p.hDrealScat, p.hDimagScat, "--output", p.psfSysReal, p.psfSysImag]
        dc10 = ["genSysPS", "--input", p.paraJSON, p.psfSysReal, p.psfSysImag, "--output", p.psSys]
        dc11 = ["genHisto", "--input", p.paraJSON, p.thetaVol, p.phiVol, p.psSys, p.hErealScat, p.hEimagScat, p.hDrealScat, p.hDimagScat, "--output", p.resAngles]
                
        # run functions
        args1 = parser.parse_args(dc1); args1.func(args1)
        args2 = parser.parse_args(dc2); args2.func(args2)
        args3 = parser.parse_args(dc3); args3.func(args3)
        args4 = parser.parse_args(dc4); args4.func(args4)
        args5 = parser.parse_args(dc5); args5.func(args5)
        args6 = parser.parse_args(dc6); args6.func(args6)
        args7 = parser.parse_args(dc7); args7.func(args7)
        args8 = parser.parse_args(dc8); args8.func(args8)
        args9 = parser.parse_args(dc9); args9.func(args9)
        args10 = parser.parse_args(dc10); args10.func(args10)
        args11 = parser.parse_args(dc11); args10.func(args11)        
        
        # time.sleep(1)
        # with open(p.paraJSON, 'r') as f:
        #      js = json.load(f)
        
    else: 
        args = parser.parse_args(); args.func(args)


# if __name__ == "__main__":
#     # 01
#     loadPara()
#     snakemake = snakemakeDebug()
#     with open(snakemake.input.paraJSON, 'r') as f:
#         jsDebzg = json.load(f)
    
#     # 02 - parallel
#     loadPadSampleVol()
#     genExcPSF()
#     genDetPSF()
#     genAngleSpace()
#     # 03 - for loop
#     #xyScan()
#     #shift vol - time consuming - rotations
#     #genPSsys - time consuming - fft
#     #map2Angle 
#     #calcAngles
#     #calcBrillo
#     #saveHisto
#     # 04 - generate results 
    
    
    