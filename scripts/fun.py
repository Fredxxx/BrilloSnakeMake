import os
import sys
from types import SimpleNamespace
import json
import argparse
import logging
import gc
import tifffile as tiff
import numpy as np
from numpy.fft import fftshift, fftfreq
from scipy.ndimage import zoom, rotate
import configparser
configparser.SafeConfigParser = configparser.ConfigParser
import biobeam as bb
import matplotlib.pyplot as plt

#%% helper functions --- not directly called by snakemake

def genPaddArray(s, vol, pad):
    
    Z, Y, X = s * pad, s * pad, s
    
    padded_scatVol = np.random.normal(1.33335, 0.00074, size=(Z, Y, X)).astype(np.float16)
    
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

def plot_max_projections(volume, voxel_size=(1.0, 1.0, 1.0), cmap='hot', title="Max Intensity Projections", space="real"):
    """
    Plottet Maximalprojektionen eines 3D-Volumes im Ortsraum (real) oder Frequenzraum (fft).
    """
    dz, dy, dx = voxel_size
    Z, Y, X = volume.shape

    # Logik für Einheiten und Skalierung
    if space == "fft":
        unit = "1/µm"
        # Frequenzschritte berechnen (df = 1 / (N * dx))
        dfz = 1.0 / (Z * dz)
        dfy = 1.0 / (Y * dy)
        dfx = 1.0 / (X * dx)
        
        # Extents für FFT (zentriert)
        extent_xy = [-X//2 * dfx, X//2 * dfx, -Y//2 * dfy, Y//2 * dfy]
        extent_xz = [-X//2 * dfx, X//2 * dfx, -Z//2 * dfz, Z//2 * dfz]
        extent_yz = [-Y//2 * dfy, Y//2 * dfy, -Z//2 * dfz, Z//2 * dfz]
    else:
        unit = "µm"
        extent_xy = [-X//2 * dx, X//2 * dx, -Y//2 * dy, Y//2 * dy]
        extent_xz = [-X//2 * dx, X//2 * dx, -Z//2 * dz, Z//2 * dz]
        extent_yz = [-Y//2 * dy, Y//2 * dy, -Z//2 * dz, Z//2 * dz]

    # Max-Projektionen berechnen
    max_xy = np.max(volume, axis=0)
    max_xz = np.max(volume, axis=1)
    max_yz = np.max(volume, axis=2)

    # Plot erstellen
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title} ({space.upper()} space)", fontsize=16)

    # XY-Projektion
    axes[0].imshow(max_xy, cmap=cmap, extent=extent_xy, origin='lower', aspect='auto')
    axes[0].set_title('Z-Projection (XY)')
    axes[0].set_xlabel(f'X ({unit})')
    axes[0].set_ylabel(f'Y ({unit})')

    # XZ-Projektion
    axes[1].imshow(max_xz, cmap=cmap, extent=extent_xz, origin='lower', aspect='auto')
    axes[1].set_title('Y-Projection (XZ)')
    axes[1].set_xlabel(f'X ({unit})')
    axes[1].set_ylabel(f'Z ({unit})')

    # YZ-Projektion
    axes[2].imshow(max_yz, cmap=cmap, extent=extent_yz, origin='lower', aspect='auto')
    axes[2].set_title('X-Projection (YZ)')
    axes[2].set_xlabel(f'Y ({unit})')
    axes[2].set_ylabel(f'Z ({unit})')

    #fig.colorbar(axes[2].images[0], ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, aspect=20)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    #plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes
  
def add_command(subParsers, name, func, parents=None):
    subparser = subParsers.add_parser(name, parents=parents or [])
    subparser.set_defaults(func=func)
    return subparser

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
    sf = 0.23/js["optExc"]["d"]
    scale_factors = (sf, sf, 4*sf)
    scatVol = zoom(scatVol, scale_factors, order=1)  # order=1 = linear interpolation
    
    if js["adv"]["showImg"] == 2:
        plot_max_projections(scatVol, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="scatVol", space="real")
        
    scatVol = genPaddArray(js["optExc"]["N"], scatVol, js["adv"]["pad"]) 
    
    if js["adv"]["showImg"] in [1, 2]:
        plot_max_projections(scatVol, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="scatVol (padded)", space="real")
    
    tiff.imwrite(outPath, scatVol)
    #print(f"--- sample loaded: {js['scatPath']}", flush=True)
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
        _ ,psfE, _, _ = bb.focus_field_beam(**params)
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
        _ ,psfE, _, _ = bb.focus_field_beam(**params)
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
        _ ,psfE, _, _ = bb.focus_field_cylindrical(**params)
    else:
        psfE = np.zeros((js["optExc"]["N"],) * 3)
    
    if js["adv"]["showImg"] == 1:
        plot_max_projections(np.abs(psfE), voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf excitation", space="real")
    elif js["adv"]["showImg"] == 2:
        plot_max_projections(np.abs(psfE), voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf excitation", space="real")
        plot_max_projections(np.angle(psfE), voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="ps excitation", space="ftt") 
        
    tiff.imwrite(outPreal, psfE.real.astype(np.float32))
    tiff.imwrite(outPimag, psfE.imag.astype(np.float32))
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
    _ ,psfD, _, _ = bb.focus_field_beam(**params)
    
    if js["adv"]["showImg"] == 1:
        plot_max_projections(np.abs(psfD), voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf detection", space="real")
    elif js["adv"]["showImg"] == 2:
        plot_max_projections(np.abs(psfD), voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="psf detection", space="real")
        plot_max_projections(np.angle(psfD), voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="ps detection", space="ftt")   

    tiff.imwrite(outPreal, psfD.real.astype(np.float32))
    tiff.imwrite(outPimag, psfD.imag.astype(np.float32))
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
        plot_max_projections(theta, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="theta vol", space="real")
    tiff.imwrite(outPtheta, theta.astype(np.float32))
    del theta, KZ, k_mag
    gc.collect()
    
    # Phi - less interesting (radial symmetric), azimutal
    phi = np.rad2deg(np.arctan2(KY, KX))       # azimuthal angle
    if js["adv"]["showImg"] == 2:
        plot_max_projections(phi, voxel_size=(js["optExc"]["N"],) * 3, cmap='hot', title="theta vol", space="real")
    tiff.imwrite(outPphi, phi.imag.astype(np.float32))
    
    print("----- angle space generated -----", flush=True)
    
def genIDXs(args):
    inPath = args.input[0]
    outPath = args.output[0]
    
    with open(inPath, 'r') as f:
        js = json.load(f)

    s = js["optExc"]["N"]
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
        start_x = round(s - xrange/2) + i * xstepSize
        end_x = start_x + s
        for j in range(ysteps):
            start_y = round(s - yrange/2) + j * ystepSize
            end_y = start_y + s
            for w in range(zsteps):
                start_z = 0 # xxx not sure but works
                end_z = start_z + s
                
                coo = [start_z, end_z, start_y, end_y, start_x, end_x]
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

def testWildcard(args):
    inJSON = args.input[0]
    inScanPar = args.input[1] 
    outPath = args.output[0]
    
    with open(inJSON, 'r') as f:
        js = json.load(f)
        
    with open(inScanPar, 'r') as f:
        scanPara = json.load(f) 
    
    with open(outPath, "w", encoding="utf-8") as dat:
        dat.write("curIDX")
        
    
    with open(outPath, "w", encoding="utf-8") as dat:
        dat.write(f"index: {outPath}")

# def propVol(i):
#     snakemake = snakemakeDebug()
#     inJSON = snakemake.input.paraJSON
#     inScanPar = snakemake.input.scanPara
#     proVol = snakemake.output.proVol
#     psfE = snakemake.output.psfE
#     psfD = snakemake.output.psfD
#     outPath = snakemake.output.psfSys
     
#     with open(inJSON, 'r') as f:
#         js = json.load(f)
#     with open(inScanPar, 'r') as f:
#         sp = json.load(f)
        
#     t = proVol[sp["coo"][i]]
#     del proVol
#     gc.collect()
    
#     t = bb.Bpm3d(dn=t, units = (js["optExc"]["d"],)*3, lam=js["optExc"]["lam"]/js["optExc"]["n0"])
#     psfEscat = t.propagate(u0 = psfE[0,:,:])

#     t = np.rot90(t, k=1, axes=(1, 2))
#     t = bb.Bpm3d(dn=t, units = (js["optExc"]["d"],)*3, lam=js["optDet"]["lam"]/js["optExc"]["n0"])
#     psfDscat = t.propagate(u0 = psfD[0,:,:])
    
#     # Powerspectrum 
#     psfS = psfEscat * psfDscat
    
    #td = padded_scatVol[coo[4]:coo[5], coo[0]:coo[1], coo[2]:coo[3]]
    #plot_max_projections(te, voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="te")
    #plot_max_projections(td, voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="td")

    # te = bb.Bpm3d(dn=te, units = (optDet.dx,)*3, lam=optDet.lam/optExc.n0)
    # psfEscat = te.propagate(u0 = psfE[0,:,:])
    # #plot_max_projections(np.abs(psfEscat), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psfEScat")
    
    # del psfE 
    # del te 
    # gc.collect()
    
    # # Detection: shift volume, init propagator, propagate and rotate
    # # t = padded_scatVol[coo[4]:coo[5], coo[0]:coo[1], coo[2]:coo[3]]
    # # t = bb.Bpm3d(dn=t, units = (optDet.dx,)*3, lam=optDet.lam/optExc.n0)
    
    # td = bb.Bpm3d(dn=td, units = (optDet.dx,)*3, lam=optDet.lam/optExc.n0)
    # psfDscat = td.propagate(u0 = psfDgen[0,:,:])
    # #plot_max_projections(np.abs(psfDscat), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psfDScat")
    
    # #psfDscat = rotPSF(psfDscat, 90)
    # psfEscat = rotPSF(psfEscat, 90)

    # del psfDgen
    # del td
    # gc.collect()
    
    # # Powerspectrum 
    # psfS = psfEscat * psfDscat
        
#%% main   
if __name__ == "__main__":

    # general io parser
    ioParser = argparse.ArgumentParser(add_help=False)
    ioParser.add_argument("--input", nargs="+", required=True)
    ioParser.add_argument("--output", nargs="+", required=True)

    # mainparser
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda args: parser.print_help())

    subParsers = parser.add_subparsers(required=True)
    
    # add subcommand - function
    add_command(subParsers, "loadPara", loadPara, parents=[ioParser])
    add_command(subParsers, "loadPadSampleVol", loadPadSampleVol, parents=[ioParser])
    add_command(subParsers, "genExcPSF", genExcPSF, parents=[ioParser])
    add_command(subParsers, "genDetPSF", genDetPSF, parents=[ioParser])
    add_command(subParsers, "genAngleSpace", genAngleSpace, parents=[ioParser])
    add_command(subParsers, "genIDXs", genIDXs, parents=[ioParser])
    add_command(subParsers, "testWildcard", testWildcard, parents=[ioParser])
    
    if len(sys.argv) <= 1:
        print("---- debug mode ----")
        p = SimpleNamespace(
                paraTXT = "../data/para.txt",
                paraJSON = "../results/01_paraTemp.json",
                propVol  = "../results/02_propVol.tif",
                psfEreal = "../results/02_psfEreal.tif",
                psfEimag = "../results/02_psfEimag.tif",
                psfDreal = "../results/02_psfDreal.tif",
                psfDimag = "../results/02_psfDimag.tif",
                thetaVol = "../results/02_thetaVol.tif",
                phiVol   = "../results/02_phiVol.tif",
                scanPara = "../results/02_scanPara.json",
                testW = "../results/03_testW_{idx}.txt"
                #psfSys   = "../results/03_psfSys_{idx}.tif"
             )
        
        # run functions
        dc1 = ["loadPara", "--input", p.paraTXT, "--output", p.paraJSON]
        dc2 = ["loadPadSampleVol", "--input", p.paraJSON, "--output", p.propVol]
        dc3 = ["genExcPSF", "--input", p.paraJSON, "--output", p.psfEreal, p.psfEimag]
        dc4 = ["genDetPSF", "--input", p.paraJSON, "--output", p.psfDreal, p.psfDimag]
        dc5 = ["genAngleSpace", "--input", p.paraJSON, "--output", p.thetaVol, p.phiVol]
        dc6 = ["genIDXs", "--input", p.paraJSON, "--output", p.scanPara]
        #dc7 = ["testWildcard", "--input", p.paraJSON, p.scanPara, "--output", p.testW]
        
        args1 = parser.parse_args(dc1); args1.func(args1)
        args2 = parser.parse_args(dc2); args2.func(args2)
        args3 = parser.parse_args(dc3); args3.func(args3)
        args4 = parser.parse_args(dc4); args4.func(args4)
        args5 = parser.parse_args(dc5); args5.func(args5)
        args6 = parser.parse_args(dc6); args6.func(args6)
        #args7 = parser.parse_args(dc7); args7.func(args7)
       
        # fakes a loop thorugh pixel
        with open(p.scanPara, 'r') as f:
            jsScanDebug = json.load(f)
        
        for curIDX in range(jsScanDebug["idxMax"]):
            print(f"--- Debugging Wildcard Index: {curIDX} ---")
            current_output = p.testW.format(idx=curIDX)
            dc_loop = ["testWildcard", "--input", p.paraJSON, p.scanPara, "--output", current_output]
            args_loop = parser.parse_args(dc_loop); args_loop.func(args_loop)
        
    else: 
        args = parser.parse_args(); args.func(args)
 
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("command", nargs="?")      
    # parser.add_argument("--input", nargs="+") 
    # parser.add_argument("--output", nargs="+")
    # args = parser.parse_args()

    # if args.command == "loadPara":
    #     loadPara(args.input[0], args.output[0])
    # elif args.command == "loadPadSampleVol":
    #     loadPadSampleVol(args.input[0], args.output[0])
    # elif args.command == "genExcPSF":
    #     realPath, imagPath = args.output 
    #     genExcPSF(args.input[0], realPath, imagPath)
    # elif args.command == "genDetPSF":
    #     realPath, imagPath = args.output 
    #     genDetPSF(args.input[0], realPath, imagPath)
    # elif args.command == "genAngleSpace":
    #     thetaVol, phiVol = args.output 
    #     genAngleSpace(args.input[0], thetaVol, phiVol)  
    # elif args.command == "genIDXs":
    #     genIDXs(args.input[0], args.output[0])
    # elif args.command == "testWildcard":
    #     testWildcard(args.input[0], args.input[1], args.output[0])
    # else: 
    #     print("----- debug modus -----")
    #     p = SimpleNamespace(
            # paraTXT = "../data/para.txt",
            # paraJSON = "../results/01_paraTemp.json",
            # propVol  = "../results/02_propVol.tif",
    #         paraTXT  = r"C:\temp\snakemake\data\para.txt",
    #         paraJSON = r"C:\temp\snakemake\results\01_paraTemp.json",
    #         propVol  = r"C:\temp\snakemake\results\02_propVol.tif",
    #         psfEreal = r"C:\temp\snakemake\results\02_psfEreal.tif",
    #         psfEimag = r"C:\temp\snakemake\results\02_psfEimag.tif",
    #         psfDreal = r"C:\temp\snakemake\results\02_psfDreal.tif",
    #         psfDimag = r"C:\temp\snakemake\results\02_psfDimag.tif",
    #         thetaVol = r"C:\temp\snakemake\results\02_thetaVol.tif",
    #         phiVol   = r"C:\temp\snakemake\results\02_phiVol.tif",
    #         scanPara = r"C:\temp\snakemake\results\02_scanPara.json",
    #         psfSys   = r"C:\temp\snakemake\results\03_psfSys_{idx}.tif"
    #     )
    #     # 01
    #     loadPara(p.paraTXT, p.paraJSON)
    #      # snakemake = snakemakeDebug()
    #      # with open(snakemake.input.paraJSON, 'r') as f:
    #      #     jsDebug = json.load(f)
        
    #     # 02 - parallel
    #     loadPadSampleVol(p.paraJSON, p.propVol)
    #     genExcPSF(p.paraJSON, p.psfEreal, p.psfEimag)
    #     genDetPSF(p.paraJSON, p.psfDreal, p.psfDimag)
    #     genAngleSpace(p.paraJSON, p.thetaVol, p.phiVol)
    #     genIDXs(p.paraJSON, p.scanPara)

    #     # # 03 run loop
    #     # with open(snakemake.input.scanPara, 'r') as f:
    #     #     jsScanDebug = json.load(f)
    #     # for curIDX in range(jsScanDebug["idxMax"]):
    #     #     testWildcard(curIDX)
    #     # # 03 run loopy
    #     # with open(snakemake.input.scanJSON, 'r') as f:
    #     #     jsScanDebug = json.load(f)
    #     # for curIDX in range(jsScanDebug["scan"]["maxIDX"]):
    #     #     r1(curIDX)
    #     #     r2(curIDX)
    #     #     r3(curIDX)
    #     #     r4(curIDX)
        
 
        
 
    
# def process_shift2(coo, padded_scatVol, psfE, psfDgen, optExc, optDet, optGen, path, theta, phi, i, j, w, idx, idxMax):
      
#    # report progress
#    per = idx/idxMax*100
#    safe_print_progress(per, idx, idxMax)
   
#    # Excitation: shift volume, init propagator and propagate
#    te = padded_scatVol[coo[4]:coo[5], coo[2]:coo[3], coo[0]:coo[1]]
#    #td=te
#    #td = np.swapaxes(te, 0, 2)

#    td = np.rot90(te, k=1, axes=(1, 2))
#    #td = padded_scatVol[coo[4]:coo[5], coo[0]:coo[1], coo[2]:coo[3]]
#    #plot_max_projections(te, voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="te")
#    #plot_max_projections(td, voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="td")

#    del padded_scatVol 
#    te = bb.Bpm3d(dn=te, units = (optDet.dx,)*3, lam=optDet.lam/optExc.n0)
#    psfEscat = te.propagate(u0 = psfE[0,:,:])
#    #plot_max_projections(np.abs(psfEscat), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psfEScat")
   
#    del psfE 
#    del te 
#    gc.collect()
   
#    # Detection: shift volume, init propagator, propagate and rotate
#    # t = padded_scatVol[coo[4]:coo[5], coo[0]:coo[1], coo[2]:coo[3]]
#    # t = bb.Bpm3d(dn=t, units = (optDet.dx,)*3, lam=optDet.lam/optExc.n0)
   
#    td = bb.Bpm3d(dn=td, units = (optDet.dx,)*3, lam=optDet.lam/optExc.n0)
#    psfDscat = td.propagate(u0 = psfDgen[0,:,:])
#    #plot_max_projections(np.abs(psfDscat), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psfDScat")
   
#    #psfDscat = rotPSF(psfDscat, 90)
#    psfEscat = rotPSF(psfEscat, 90)

#    del psfDgen
#    del td
#    gc.collect()
   
#    # Powerspectrum 
#    psfS = psfEscat * psfDscat
#    #plot_max_projections(np.abs(psfS), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psfS")
#    #psS = fftcpuPS(psfS)
#    #plot_max_projections(psS, voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psfS")
   
   
#    # calc results
#    resS = calcMain(fftgpuPS(psfS), 'sys', theta, phi, optDet, optGen, idx, coo, i, j, w, path)
   
#    del psfS
#    gc.collect()
#    saveHisto(resS, path, optDet)
   
#    # resE = calcMain(fftcpuPS(psfEscat), 'exc', theta, phi, optDet, optGen, idx, coo, i, j, w, path)
#    # del psfEscat
#    # gc.collect()
#    # saveHisto(resE, path, optDet)
   
#    # resD = calcMain(fftcpuPS(psfDscat), 'det', theta, phi, optDet, optGen, idx, coo, i, j, w, path)
#    # del psfDscat
#    # gc.collect()
#    # saveHisto(resD, path, optDet)
    
 
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
    
    
    