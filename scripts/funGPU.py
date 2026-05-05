import json
import argparse
import gc
import os
import tifffile as tiff # type: ignore
import numpy as np # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable # type: ignore
import configparser
configparser.SafeConfigParser = configparser.ConfigParser
import biobeam as bb # type: ignore
import matplotlib.pyplot as plt # type: ignore

#%% helper functions --- not directly called by snakemake

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

def test(args):

    ## def args
    inPath = args.input[0]
    outPath = args.output[0]
    print("----- test started -----", flush=True)
    print("input:", inPath, flush=True)
    print("output:", outPath, flush=True)
    print("done", flush=True)
    #print("OUTPATH:", outPath, flush=True)
    t = np.zeros((3, 3))
    #os.makedirs(os.path.dirname(outPath), exist_ok=True)
    tiff.imwrite(outPath, t)
    print("----- test finished -----", flush=True)

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

def propExcVol(args):
    # def args
    inPath = args.input[0]
    inPSFreal = args.input[1]
    inPSFimag = args.input[2]
    inPropVol = args.input[3]
    outPSFreal = args.output[0]
    outPSFimag = args.output[1]
    
    # load general + scan parameters
    with open(inPath, 'r') as f:
        js = json.load(f)
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
    inPSFreal = args.input[1]
    inPSFimag = args.input[2]
    inPropVol = args.input[3]
    outPSFreal = args.output[0]
    outPSFimag = args.output[1]
    
    # load general + scan parameters
    with open(inPath, 'r') as f:
        js = json.load(f)
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
    add_command(subParsers, "test", test, parents=[ioParser])
    add_command(subParsers, "genExcField", genExcField, parents=[ioParser])
    add_command(subParsers, "genDetField", genDetField, parents=[ioParser])
    add_command(subParsers, "propExcVol", propExcVol, parents=[ioParser])
    add_command(subParsers, "propDetVol", propDetVol, parents=[ioParser])
    
    args = parser.parse_args()
    args.func(args) 