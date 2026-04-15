import os
#import time
import sys
import json
import argparse
#import logging
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
    # 
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

    _ = axes[0].imshow(max_xy, cmap=cmap, extent=extent_xy, origin='lower',
                         aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title('Z-Projection (XY)')
    axes[0].set_xlabel(f'X ({unit})')
    axes[0].set_ylabel(f'Y ({unit})')

    _ = axes[1].imshow(max_xz, cmap=cmap, extent=extent_xz, origin='lower',
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

def saveHistPlot(res, h, path, a):
    
    _, theMu, theSig = res.theFit
    _, phiMu, phiSig = res.phiFit
    
    fig = plt.figure(figsize=(8, 8))

    ax_main  = fig.add_axes([0.1, 0.1, 0.6, 0.6])
    ax_top   = fig.add_axes([0.1, 0.72, 0.6, 0.18], sharex=ax_main)
    ax_right = fig.add_axes([0.72, 0.1, 0.18, 0.6], sharey=ax_main)

    # Main image
    _ = ax_main.imshow(
        res.hist,
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

def deg2bs(js, deg):
    q0 = (1/js["optDet"]["lam"])**2
    f = q0 * js["brillo"]["Vs"]
    return f*np.sin(np.deg2rad(deg)/2) * 10**-9

def genIDXs(js):
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
    coorSets = []
    xIdx = []
    yIdx = []
    zIdx = []
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
                coorSets.append(coo)
                xIdx.append(i)
                yIdx.append(j)
                zIdx.append(w)

    # constitute scanParas and save as json
    scanData = {
        "idxMax": len(coorSets),
        "idxVector": list(range(len(coorSets))),
        "x": {"steps": xsteps, "range": xrange, "stepSize": xstepSize, "idx": xIdx},
        "y": {"steps": ysteps, "range": yrange, "stepSize": ystepSize, "idx": yIdx},
        "z": {"steps": zsteps, "range": zrange, "stepSize": zstepSize, "idx": zIdx},
        "coo": coorSets
    }
    return scanData

#%% active functions -- which are called by snakemake

def loadPadSampleVol(args):
    
    # def args
    inPath = args.input[0]
    outPath = args.output[0]
    
    # load general parameters
    with open(inPath, 'r') as f:
        js = json.load(f)
        
    # load image, swap axis, zoom
    scatVol = tiff.imread(js["scatPath"])/10000 + js["adv"]["nOff"]
    scatVol = np.swapaxes(scatVol, 0, 2)
    #sf = 0.23/js["optExc"]["d"]
    sf = 0.23*4//js["optExc"]["d"]
    scale_factors = (sf, sf, 4*sf)
    scatVol = zoom(scatVol, scale_factors, order=1)  # order=1 = linear interpolation
    
    # plot projections
    if js["adv"]["showImg"] == 2:
        plot_max_projections(scatVol, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="scatVol (loadPadSampleVol)", space="real")
    
    # pad array    
    scatVol = genPaddArray(js["optExc"]["N"], scatVol, js["adv"]["pad"]) 
    
    # plot projections
    if js["adv"]["showImg"] in [1, 2]:
        plot_max_projections(scatVol, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="scatVol (loadPadSampleVol)", space="real")
    
    # save volume
    tiff.imwrite(outPath, scatVol)
    print("----- sample loaded -----", flush=True)

def genExcField(args):
    
    # def args
    inPath = args.input[0]
    outHreal = args.output[0]
    outHimag = args.output[1]
    
    # load general parameters
    with open(inPath, 'r') as f:
        js = json.load(f)
    
    # generate excitation E-field (GaussBeam, BesselBeam, GaussSheet)
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
    
    # plot excitation
    if js["adv"]["showImg"] == 1:
        plot_max_projections(np.abs(hE)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf excitation intensity (genExcPSF)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])
    elif js["adv"]["showImg"] == 2:
        plot_max_projections(np.abs(hE)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf excitation intensity (genExcPSF)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])
        plot_max_projections(np.angle(hE), voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf excitation phase (genExcPSF)", space="ftt", globalPlotScaling=js["adv"]["globalPlotScaling"]) 
        
    # save real and imag part of E-field
    tiff.imwrite(outHreal, hE.real.astype(np.float32))
    tiff.imwrite(outHimag, hE.imag.astype(np.float32))
    print("----- excitation field generated -----", flush=True) 

def genDetField(args):  
    inPath = args.input[0]
    outHreal = args.output[0]
    outHimag = args.output[1]
    
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

    tiff.imwrite(outHreal, hD.real.astype(np.float32))
    tiff.imwrite(outHimag, hD.imag.astype(np.float32))
    print("----- detection field generated -----", flush=True)

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
    
# def genIDXs(args):
#     inPath = args.input[0]
#     outPath = args.output[0]
    
#     with open(inPath, 'r') as f:
#         js = json.load(f)

#     sPad = js["optExc"]["N"]*js["adv"]["pad"]
#     sVol = js["optExc"]["N"]
    
#     xsteps = js["scanPara"]["xSteps"]
#     xrange = js["scanPara"]["xRange"]
#     ysteps = xsteps 
#     yrange = xrange
#     zsteps = 1
#     zrange = xrange

#     # calculate step size
#     xstepSize = 0 if xsteps == 1 else round(xrange / (xsteps - 1))
#     ystepSize = 0 if ysteps == 1 else round(yrange / (ysteps - 1))
#     zstepSize = 0 if zsteps == 1 else round(zrange / (zsteps - 1))

#     # set scan coordinates
#     coorSets = []
#     xIdx = []
#     yIdx = []
#     zIdx = []
#     for i in range(xsteps):
#         start_x = round(sPad/2 - (sVol+xrange)/2) + i * xstepSize
#         end_x = start_x + sVol
#         for j in range(ysteps):
#             start_y = round(sPad/2 - (sVol+yrange)/2) + j * ystepSize
#             end_y = start_y + sVol
#             for w in range(zsteps):
#                 start_z = 0 # xxx not sure but works
#                 end_z = start_z + sVol
                
#                 coo = [start_y, end_y, start_x, end_x, start_z, end_z, ]
#                 coorSets.append(coo)
#                 xIdx.append(i)
#                 yIdx.append(j)
#                 zIdx.append(w)

#     # constitute scanParas and save as json
#     scanData = {
#         "idxMax": len(coorSets),
#         "idxVector": list(range(len(coorSets))),
#         "x": {"steps": xsteps, "range": xrange, "stepSize": xstepSize, "idx": xIdx},
#         "y": {"steps": ysteps, "range": yrange, "stepSize": ystepSize, "idx": yIdx},
#         "z": {"steps": zsteps, "range": zrange, "stepSize": zstepSize, "idx": zIdx},
#         "coo": coorSets
#     }
#     os.makedirs(os.path.dirname(outPath), exist_ok=True)
#     with open(outPath, 'w') as f:
#         json.dump(scanData, f, indent=4)
#     print("----- scan para prepared -----", flush=True)

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
    #with open(inScanPar, 'r') as f:
    #    sp = json.load(f)
    sp = genIDXs(js)
    i = int(outPSFreal.split("_")[-1].split(".")[0]) # identify idx    
        
    # load volumes and make complex field    
    psfReal = tiff.imread(inPSFreal)
    psfImag = tiff.imread(inPSFimag)
    proVol = tiff.imread(inPropVol)
    psf = psfReal + 1j*psfImag 
    
    # propaget though medium
    t = proVol[sp["coo"][i][0]:sp["coo"][i][1], sp["coo"][i][2]:sp["coo"][i][3], sp["coo"][i][4]:sp["coo"][i][5]]
    #os.makedirs("results/deb", exist_ok=True)
    #tiff.imwrite(f"results/deb/propExc_t_{i}.tif", t) # xxx deb
    del proVol; gc.collect()
    t = bb.Bpm3d(dn=t, units = (js["optExc"]["d"],)*3, lam=js["optExc"]["lam"]/js["optExc"]["n0"])
    psfScat = t.propagate(u0 = psf[0,:,:]) 
    del psf, t; gc.collect()
    
    # rotate
    if js["optDet"]["angle"] == 0:
        psfScat = np.rot90(psfScat, k=1, axes=(1, 2))
    elif js["optDet"]["angle"] != 90:
        print("----- error propExcVol: optDet.angle only supported for 0 and 90. Continue with 0 case. -----", flush=True)
    #tiff.imwrite(f"results/deb/propExc_psfScat_{i}.tif", np.abs(psfScat)**2) # xxx deb
    
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
    #with open(inScanPar, 'r') as f:
    #    sp = json.load(f)
    sp = genIDXs(js)
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
        #tiff.imwrite(f"results/deb/propDet_t_{i}.tif", t) # xxx deb
        t = bb.Bpm3d(dn=t, units = (js["optExc"]["d"],)*3, lam=js["optDet"]["lam"]/js["optExc"]["n0"])
        psfScat = t.propagate(u0 = psf[0,:,:])
        del psf, t; gc.collect()    
        psfScat = np.rot90(psfScat, k=1, axes=(1, 2))
    elif js["optDet"]["angle"] == 90:
        t = proVol[sp["coo"][i][0]:sp["coo"][i][1], sp["coo"][i][2]:sp["coo"][i][3], sp["coo"][i][4]:sp["coo"][i][5]]
        t = np.rot90(t, k=1, axes=(1, 2))
        del proVol; gc.collect()
        #tiff.imwrite(f"results/deb/propDet_t_{i}.tif", t) # xxx deb
        t = bb.Bpm3d(dn=t, units = (js["optExc"]["d"],)*3, lam=js["optDet"]["lam"]/js["optExc"]["n0"])
        psfScat = t.propagate(u0 = psf[0,:,:])
        del psf, t; gc.collect() 
        psfScat = np.rot90(psfScat, k=3, axes=(1, 2)) # rotates 90 back (270°)
    else:
        print("----- error propExcVol: optDet.angle only supported for 0 and 90. Continue with 0 case. -----", flush=True)
    #tiff.imwrite(f"results/deb/propDet_psfScat_{i}.tif", np.abs(psfScat)**2) # xxx deb
    
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
    excHreal = tiff.imread(inPSFrealExc)
    excHimag = tiff.imread(inPSFimagExc)
    
    excH = excHreal + 1j*excHimag
    del excHreal, excHimag; gc.collect()
    
    # load volumes and make complex field (detection)  
    detHreal = tiff.imread(inPSFrealDet)
    detHimag = tiff.imread(inPSFimagDet)
    detH = detHreal + 1j*detHimag
    del detHreal, detHimag; gc.collect()
    
    # ToDo implement coherent, incoherent scattering here
    # gen system PSF 
    sysH = excH * detH
    
    # plotting
    if js["adv"]["showImg"] == 1:
        plot_max_projections(np.abs(sysH)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf system intensity (genPS)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])    
    
    # save 
    tiff.imwrite(outPSFreal, sysH.real.astype(np.float32))
    tiff.imwrite(outPSFimag, sysH.imag.astype(np.float32))
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
    
    # ToDo implement OTF and MTF 
    psf = psfReal + 1j*psfImag
    del psfReal, psfImag; gc.collect()
    
    # calc power spectrum
    ps = fftcpuPS(psf)
    #tiff.imwrite(f"results/deb/genSysPS_ps_{i}.tif", ps) # xxx deb
    
    # plotting
    if js["adv"]["showImg"] == 1:
        plot_max_projections(np.abs(ps)**2, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="power spectrum (genPS)", space="real", globalPlotScaling=js["adv"]["globalPlotScaling"])    
    
    # save
    tiff.imwrite(outPS, ps)
    print(f"----- ps sys generated idx={i} -----", flush=True)
        
def genHisto(args):
    
    # adjust input depending on mode
    if args.mode == "sys":
        inSysPS = args.input[3]
        ps = tiff.imread(inSysPS)
        calcHisto(args, ps)
    elif args.mode in ("exc", "det"):
        inPSFreal = args.input[3]
        inPSFimag = args.input[4]
        psfReal = tiff.imread(inPSFreal)
        psfImag = tiff.imread(inPSFimag)
        psf = psfReal + 1j*psfImag
        ps = fftcpuPS(psf)
        calcHisto(args, ps)
    else:
        raise ValueError("-- error in genHisto: This should not happen! --")
  
def calcHisto(args, ps): 
      
    # def args
    inPath = args.input[0]
    inThetaVol = args.input[1]
    inPhiVol = args.input[2]
    outResAngle = args.output[0]
    mode = args.mode
    
    # load general parameters
    with open(inPath, 'r') as f:
        js = json.load(f)
    i = int(outResAngle.split("_")[-1].split(".")[0])
    res = {}
    h = {}
    
    # load volume and bins
    theta = tiff.imread(inThetaVol)
    phi = tiff.imread(inPhiVol)
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
        hist = hist_sum / counts
        hist[counts == 0] = 0
    
    # calc space center of mass
    com = tuple(np.round(center_of_mass(ps)).astype(int))
    res["ComPix"] = com
    res["degComTheta"] = float(theta[com])
    res["degComPhi"] = float(phi[com])
    
    #calc space variance
    coords = np.indices(ps.shape).reshape(3, -1)
    with np.errstate(divide='ignore', invalid='ignore'):
        res["degVarTheta"] = np.average((thetaFlat - theta[com])**2, weights=psFlat)
        res["degVarPhi"]   = np.average((phiFlat   - phi[com])**2,   weights=psFlat)
        res["pixVarXYZ"] = np.average((coords.T - com)**2, axis=0, weights=psFlat)
        res["pixVar"] = np.sqrt(np.sum(res["pixVarXYZ"]))

    h["theC"] = 0.5 * (theBins[:-1] + theBins[1:])
    h["phiC"]   = 0.5 * (phiBins[:-1] + phiBins[1:]) 
    h["theProj"] = hist.T.sum(axis=0)
    h["phiProj"] = hist.T.sum(axis=1)

    try:
        res["degTheFit"], _ = fitGauss(h["theC"], h["theProj"])
    except RuntimeError:  
        print(f"-- Warning: Gauss fit did not converge for theta idx={i} mode={mode} --", flush=True)
        res["degTheFit"] = None
    try:    
        res["degPhiFit"], _ = fitGauss(h["phiC"], h["phiProj"])
    except RuntimeError:  
        print(f"-- Warning: Gauss fit did not converge for phi idx={i} mode={mode} --", flush=True)
        res["degPhiFit"] = None
            
    res["hist"] = hist
    # save
    if js["calc"]["fig"] == 1 and res["degTheFit"] is not None and res["degPhiFit"] is not None:
        path = f"../results/hist_{mode}_{i}.tif"
        saveHistPlot(res, h, path, js["optDet"]["angle"])
    os.makedirs(os.path.dirname(outResAngle), exist_ok=True)
    with open(outResAngle, 'w') as f:
        json.dump(res, f, indent=4, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    print(f"----- angle histo generated idx={i} mode={mode}-----", flush=True)
    
def calcBrillo(args):
    # def args
    inPath = args.input[0]
    inResDeg = args.input[1]
    outResBS = args.output[0]
    
    # load general parameters
    with open(inPath, 'r') as f:
        js = json.load(f)
    with open(inResDeg, 'r') as f:
        res = json.load(f)
        
    parts = os.path.splitext(os.path.basename(outResBS))[0].split("_")
    mode, i = parts[-2], int(parts[-1])

    res["bsComTheta"] = deg2bs(js, res["degComTheta"])
    res["bsComPhi"] = deg2bs(js, res["degComPhi"])
    res["bsVarTheta"] = deg2bs(js, res["degVarTheta"])
    res["bsVarPhi"] = deg2bs(js, res["degVarPhi"])
    if res["degTheFit"] is not None:
        res["bsTheFit"] = [deg2bs(js, d) for d in res["degTheFit"]]
    else:
        res["bsTheFit"] = None
    if res["degPhiFit"] is not None:
        res["bsPhiFit"] = [deg2bs(js, d) for d in res["degPhiFit"]]
    else:
        res["bsPhiFit"] = None
    
    # move histo to end of json - only cosmetics when looking at json file directly
    res["hist"] = res.pop("hist", None)
    
    with open(outResBS, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"----- converted to brillo spec idx={i} mode={mode}-----", flush=True)

def constImag(args):
    # def args
    inPath = args.input[0]
    #inScan = args.input[1]
    inRes = args.input[2:]
    outImg = args.output[0]
    mode = args.mode
    
    # # load general parameters
    # with open(inPath, 'r') as f:
    #     js = json.load(f)
    # load scan parameters
    #with open(inScan, 'r') as f:
    #    scan = json.load(f)
    scan = genIDXs(inPath)

    
    os.makedirs(outImg, exist_ok=True)
    
    mL1 = ["degComTheta", "degComPhi", "degVarTheta", "degVarPhi", 
           "bsComTheta", "bsComPhi", "bsVarTheta", "bsVarTheta", "pixVar"]
    for m in mL1:
        img = np.zeros((scan["x"]["steps"], scan["y"]["steps"], scan["z"]["steps"]), dtype=np.float32)
        for i, p in enumerate(inRes):
            with open(p, 'r') as f:
                res = json.load(f)
            img[scan["x"]["idx"][i], scan["y"]["idx"][i], scan["z"]["idx"][i]] = res[m]
        pathOut = outImg + "/" + m + ".tif"    
        tiff.imwrite(pathOut, img)
    
    mL2 = ["degTheFit", "bsTheFit"]
    for m in mL2:
        img = np.zeros((scan["x"]["steps"], scan["y"]["steps"], scan["z"]["steps"]), dtype=np.float32)
        for i, p in enumerate(inRes):
            with open(p, 'r') as f:
                res = json.load(f)
            img[scan["x"]["idx"][i], scan["y"]["idx"][i], scan["z"]["idx"][i]] = res[m][1]
        pathOut = outImg + "/" + m + "_mu.tif"    
        tiff.imwrite(pathOut, img)
        
    for m in mL2:
        img = np.zeros((scan["x"]["steps"], scan["y"]["steps"], scan["z"]["steps"]), dtype=np.float32)
        for i, p in enumerate(inRes):
            with open(p, 'r') as f:
                res = json.load(f)
            img[scan["x"]["idx"][i], scan["y"]["idx"][i], scan["z"]["idx"][i]] = res[m][2]
        pathOut = outImg + "/" + m + "_sig.tif"    
        tiff.imwrite(pathOut, img)
    print(f"----- constitute image mode={mode} -----", flush=True)
    
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
    add_command(subParsers, "loadPadSampleVol", loadPadSampleVol, parents=[ioParser])
    add_command(subParsers, "genExcField", genExcField, parents=[ioParser])
    add_command(subParsers, "genDetField", genDetField, parents=[ioParser])
    add_command(subParsers, "genAngleSpace", genAngleSpace, parents=[ioParser])
    add_command(subParsers, "genIDXs", genIDXs, parents=[ioParser])
    add_command(subParsers, "propExcVol", propExcVol, parents=[ioParser])
    add_command(subParsers, "propDetVol", propDetVol, parents=[ioParser])
    add_command(subParsers, "genSysPSF", genSysPSF, parents=[ioParser])
    add_command(subParsers, "genSysPS", genSysPS, parents=[ioParser])
    add_command(subParsers, "genHisto", genHisto, parents=[ioParser]).add_argument("--mode", default="sys", choices=["sys","exc","det"])
    add_command(subParsers, "calcBrillo", calcBrillo, parents=[ioParser])
    add_command(subParsers, "constImag", constImag, parents=[ioParser]).add_argument("--mode", default="sys", choices=["sys","exc","det"])
    
    if len(sys.argv) <= 1:
        print("----- debug mode -----")
        
        # list of paths
        p = {
            "para": "../data/para.json",
            "propVol": "../results/01_propVol.tif",
            "hEreal": "../results/01_excHreal.tif",
            "hEimag": "../results/01_excHimag.tif",
            "hDreal": "../results/01_detHreal.tif",
            "hDimag": "../results/01_detHimag.tif",
            "thetaVol": "../results/01_thetaVol.tif",
            "phiVol": "../results/01_phiVol.tif",
            "scanPara": "../results/01_scanPara.json",
            "hErealScat": "../results/02_excHrealScat_0.tif",
            "hEimagScat": "../results/02_excHimagScat_0.tif",
            "hDrealScat": "../results/02_detHrealScat_0.tif",
            "hDimagScat": "../results/02_detHimagScat_0.tif",
            "sysHReal": "../results/03_sysHReal_0.tif",
            "sysHImag": "../results/03_sysHImag_0.tif",
            "psSys": "../results/03_psSys_0.tif",
            "resDeg": "../results/04_resDeg_sys_0.json",
            "resBS": "../results/04_resBS_sys_0.json",
            "resDir": "../results/sys"
             }
             
        # define in and out
        dc1 = ["loadPadSampleVol", "--input", p["para"], "--output", p["propVol"]]
        dc2 = ["genExcField", "--input", p["para"], "--output", p["hEreal"], p["hEimag"]]
        dc3 = ["genDetField", "--input", p["para"], "--output", p["hDreal"], p["hDimag"]]
        dc4 = ["genAngleSpace", "--input", p["para"], "--output", p["thetaVol"], p["phiVol"]]
        dc5 = ["genIDXs", "--input", p["para"], "--output", p["scanPara"]]
        dc6 = ["propExcVol", "--input", p["para"], p["scanPara"], p["hEreal"],
               p["hEimag"], p["propVol"], "--output", p["hErealScat"], p["hEimagScat"]]
        dc7 = ["propDetVol", "--input", p["para"], p["scanPara"], p["hDreal"], 
               p["hDimag"], p["propVol"], "--output", p["hDrealScat"], p["hDimagScat"]]
        dc8 = ["genSysPSF", "--input", p["para"], p["hErealScat"], p["hEimagScat"], 
               p["hDrealScat"], p["hDimagScat"], "--output", p["sysHReal"], p["sysHImag"]]
        dc9 = ["genSysPS", "--input", p["para"], p["sysHReal"], p["sysHImag"], 
                "--output", p["psSys"]]
        dc10 = ["genHisto", "--input", p["para"], p["thetaVol"], p["phiVol"], 
                p["psSys"], p["hErealScat"], p["hEimagScat"], p["hDrealScat"], p["hDimagScat"], 
                "--output", p["resDeg"], "--mode", "sys"]
        dc11 = ["calcBrillo", "--input", p["para"], p["resDeg"], "--output", p["resBS"]]
        dc12 = ["constImag", "--input", p["para"],p["scanPara"], p["resBS"], "--output", p["resDir"], "--mode", "sys"]
                
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
        args11 = parser.parse_args(dc11); args11.func(args11)  
        args12 = parser.parse_args(dc12); args12.func(args12) 
        
    else: 
        args = parser.parse_args(); args.func(args)
    
    