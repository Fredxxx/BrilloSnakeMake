import json

def getMaxIDX(wildcards):
    cp = checkpoints.genIDXs.get().output.scanPara
    with open(cp, "r") as f:
        data = json.load(f)
    return data["idxVector"]

with open("data/para.json", "r") as f:
    js = json.load(f)

modes = []
if js["calc"]["sys"] == 1:
    modes.append("sys")
if js["calc"]["exc"] == 1:
    modes.append("exc")
if js["calc"]["det"] == 1:
    modes.append("det")

print(f"[DEBUG] active modes: {modes}")

rule all:
    input:
        "results/01_propVol.tif",
        "results/01_psfEreal.tif",
        "results/01_psfEimag.tif",
        "results/01_psfDreal.tif",
        "results/01_psfDimag.tif",
        "results/01_thetaVol.tif",
        "results/01_phiVol.tif",
        "results/01_scanPara.json",
        expand("results/02_psfErealScat_{idx}.tif", idx=getMaxIDX),
        expand("results/02_psfEimagScat_{idx}.tif", idx=getMaxIDX),
        expand("results/02_psfDrealScat_{idx}.tif", idx=getMaxIDX),
        expand("results/02_psfDimagScat_{idx}.tif", idx=getMaxIDX),
        expand("results/03_psfSysReal_{idx}.tif", idx=getMaxIDX),
        expand("results/03_psfSysImag_{idx}.tif", idx=getMaxIDX),
        expand("results/03_psSys_{idx}.tif", idx=getMaxIDX),
        expand("results/04_resDeg_{mode}_{idx}.json", mode=modes, idx=getMaxIDX),
        expand("results/04_resBS_{mode}_{idx}.json", mode=modes, idx=getMaxIDX),
        expand("results/fin_{mode}", mode=modes)
        #*[expand("results/04_resDeg_{mode}_{idx}.json", mode=modes, idx=getMaxIDX) for m in modes],
        #*[expand("results/04_resBS_{mode}_{idx}.json", mode=modes, idx=getMaxIDX) for m in modes],
        #*[expand("results/fin_{mode}", mode=modes) for m in modes]

rule loadPadSampleVol:
    input:
        para = "data/para.json"
    output:
        propVol = "results/01_propVol.tif"
    shell:
        """
        python scripts/fun.py loadPadSampleVol \
        --input {input.para} \
        --output {output.propVol}
        """

rule genExcPSF:
    input:
        para = "data/para.json" 
    output: 
        psfEreal = "results/01_psfEreal.tif",
        psfEimag = "results/01_psfEimag.tif"
    shell:
        """
        python scripts/fun.py genExcPSF \
        --input {input.para}  \
        --output {output.psfEreal} {output.psfEimag} 
        """

rule genDetPSF:
    input:
        para = "data/para.json"
    output: 
        psfDreal = "results/01_psfDreal.tif",
        psfDimag = "results/01_psfDimag.tif"
    shell:
        """
        python scripts/fun.py genDetPSF \
        --input {input.para} \
        --output {output.psfDreal} {output.psfDimag} 
        """

rule genAngleSpace:
    input:
        para = "data/para.json"
    output:
        thetaVol = "results/01_thetaVol.tif",
        phiVol = "results/01_phiVol.tif"
    shell:
        """
        python scripts/fun.py genAngleSpace \
        --input {input.para} \
        --output {output.thetaVol} {output.phiVol} 
        """

checkpoint genIDXs:
    input:
        para = "data/para.json"
    output:
        scanPara = "results/01_scanPara.json"
    shell:
        """
        python scripts/fun.py genIDXs \
        --input {input.para} \
        --output {output.scanPara}
        """

rule propExcVol:
    input:
        para = "data/para.json",
        scanPara = "results/01_scanPara.json",
        inPSFreal = "results/01_psfEreal.tif",
        inPSFimag = "results/01_psfEimag.tif",
        inPropVol = "results/01_propVol.tif"
    output:
        outPSFreal = temp("results/02_psfErealScat_{idx}.tif"),
        outPSFimag = temp("results/02_psfEimagScat_{idx}.tif")
    shell:
        """
        python scripts/fun.py propExcVol \
        --input {input.para} {input.scanPara} {input.inPSFreal} {input.inPSFimag} {input.inPropVol}\
        --output {output.outPSFreal} {output.outPSFimag}
        """

rule propDetVol:
    input:
        para = "data/para.json",
        scanPara = "results/01_scanPara.json",
        inPSFreal = "results/01_psfDreal.tif",
        inPSFimag = "results/01_psfDimag.tif",
        inPropVol = "results/01_propVol.tif"
    output:
        outPSFreal = temp("results/02_psfDrealScat_{idx}.tif"),
        outPSFimag = temp("results/02_psfDimagScat_{idx}.tif")
    shell:
        """
        python scripts/fun.py propDetVol \
        --input {input.para} {input.scanPara} {input.inPSFreal} {input.inPSFimag} {input.inPropVol}\
        --output {output.outPSFreal} {output.outPSFimag}
        """

rule genSysPSF:
    input:
        para = "data/para.json",
        inPSFrealExc = "results/02_psfErealScat_{idx}.tif",
        inPSFimagExc = "results/02_psfEimagScat_{idx}.tif", 
        inPSFrealDet = "results/02_psfDrealScat_{idx}.tif",
        inPSFimagDet = "results/02_psfDimagScat_{idx}.tif"
    output:
        psfSysReal = temp("results/03_psfSysReal_{idx}.tif"),
        psfSysImag = temp("results/03_psfSysImag_{idx}.tif")
    shell:
        """
        python scripts/fun.py genSysPSF \
        --input {input.para} {input.inPSFrealExc} {input.inPSFimagExc} \
                {input.inPSFrealDet} {input.inPSFimagDet}\
        --output {output.psfSysReal} {output.psfSysImag}
        """

rule genSysPS:
    input:
        para = "data/para.json",
        inPsfSysReal = "results/03_psfSysReal_{idx}.tif",
        inPsfSysImag = "results/03_psfSysImag_{idx}.tif"
    output:
        outPS = temp("results/03_psSys_{idx}.tif")
    shell:
        """
        python scripts/fun.py genSysPS \
        --input {input.para} {input.inPsfSysReal} {input.inPsfSysImag} \
        --output {output.outPS}
        """

def getGenHistoInputs(wildcards):
    #with open("data/para.json", "r") as f:
    #    js = json.load(f)

    mode = wildcards.mode
    idx = wildcards.idx

    inputs = ["data/para.json", "results/01_thetaVol.tif", "results/01_phiVol.tif"]

    if mode == "sys":
        inputs.append(f"results/03_psSys_{idx}.tif")
    elif mode == "exc":
        inputs.extend([
            f"results/02_psfErealScat_{idx}.tif",
            f"results/02_psfEimagScat_{idx}.tif"
        ])
    elif mode == "det":
        inputs.extend([
            f"results/02_psfDrealScat_{idx}.tif",
            f"results/02_psfDimagScat_{idx}.tif"
        ])

    print(f"[DEBUG] genHisto inputs for mode={mode}, idx={idx}: {inputs}")
    
    return inputs

rule genHisto:
    input:
        getGenHistoInputs
    output:
        outRes = temp("results/04_resDeg_{mode}_{idx}.json")
    shell:
        """
        python scripts/fun.py genHisto \
            --mode {wildcards.mode} \
            --input {input} \
            --output {output.outRes}
        """

rule calcBrillo:
    input:
        para = "data/para.json",
        inRes = "results/04_resDeg_{mode}_{idx}.json"
    output:
        outRes = "results/04_resBS_{mode}_{idx}.json"
    shell:
        """
        python scripts/fun.py calcBrillo \
        --input {input.para} {input.inRes} \
        --output {output.outRes}
        """


def getCIinputs(wildcards):
    outs = expand("results/04_resBS_{mode}_{idx}.json", 
                   mode=wildcards.mode, 
                   idx=getMaxIDX(wildcards))
    return outs

rule constImag:
    input:
        para = "data/para.json",
        inScan = "results/01_scanPara.json",
        inRes = getCIinputs
    output:
        outRes = directory("results/fin_{mode}")
    shell:
        """
        python scripts/fun.py constImag \
        --mode {wildcards.mode} \
        --input {input.para} {input.inScan} {input.inRes} \
        --output {output.outRes}
        """