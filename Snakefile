import json

def getMaxIDX(wildcards):
    #cp = checkpoints.genIDXs.get().output.scanPara
    with open("data/para.json", "r") as f:
        js = json.load(f)
        idxMax = js["scanPara"]["xSteps"]*js["scanPara"]["xSteps"] # make logic from pipeline
    return list(range(idxMax)) #pass List 
    #with open(cp, "r") as f:
    #    data = json.load(f)
    #return data["idxVector"]

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
        #"results/01_propVol.tif",
        #"results/01_excHreal.tif",
        #"results/01_excHimag.tif",
        #"results/01_detHreal.tif",
        #"results/01_detHimag.tif",
        #"results/01_thetaVol.tif",
        #"results/01_phiVol.tif",
        #"results/01_scanPara.json",
        #expand("results/02_excHrealScat_{idx}.tif", idx=getMaxIDX),
        #expand("results/02_excHimagScat_{idx}.tif", idx=getMaxIDX),
        #expand("results/02_detHrealScat_{idx}.tif", idx=getMaxIDX),
        #expand("results/02_detHimagScat_{idx}.tif", idx=getMaxIDX),
        #expand("results/03_sysHReal_{idx}.tif", idx=getMaxIDX),
        #expand("results/03_sysHImag_{idx}.tif", idx=getMaxIDX),
        #expand("results/03_psSys_{idx}.tif", idx=getMaxIDX),
        #expand("results/04_resDeg_{mode}_{idx}.json", mode=modes, idx=getMaxIDX),
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

rule genExcField:
    input:
        para = "data/para.json" 
    output: 
        excHreal = "results/01_excHreal.tif",
        excHimag = "results/01_excHimag.tif"
    shell:
        """
        python scripts/fun.py genExcField \
        --input {input.para}  \
        --output {output.excHreal} {output.excHimag} 
        """

rule genDetField:
    input:
        para = "data/para.json"
    output: 
        detHreal = "results/01_detHreal.tif",
        detHimag = "results/01_detHimag.tif"
    shell:
        """
        python scripts/fun.py genDetField \
        --input {input.para} \
        --output {output.detHreal} {output.detHimag} 
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

# move this logic to python internal 
#rule genIDXs:
#    input:
#        para = "data/para.json"
#    output:
#        scanPara = "results/01_scanPara.json"
#    shell:
#        """
#        python scripts/fun.py genIDXs \
#        --input {input.para} \
#        --output {output.scanPara}
#        """

rule propExcVol:
    input:
        para = "data/para.json",
        #scanPara = "results/01_scanPara.json",
        scanPara = "data/para.json",
        inPSFreal = "results/01_excHreal.tif",
        inPSFimag = "results/01_excHimag.tif",
        inPropVol = "results/01_propVol.tif"
    output:
        outPSFreal = temp("results/02_excHrealScat_{idx}.tif"),
        outPSFimag = temp("results/02_excHimagScat_{idx}.tif")
    shell:
        """
        python scripts/fun.py propExcVol \
        --input {input.para} {input.scanPara} {input.inPSFreal} {input.inPSFimag} {input.inPropVol}\
        --output {output.outPSFreal} {output.outPSFimag}
        """

rule propDetVol:
    input:
        para = "data/para.json",
        #scanPara = "results/01_scanPara.json",
        scanPara = "data/para.json",
        inPSFreal = "results/01_detHreal.tif",
        inPSFimag = "results/01_detHimag.tif",
        inPropVol = "results/01_propVol.tif"
    output:
        outPSFreal = temp("results/02_detHrealScat_{idx}.tif"),
        outPSFimag = temp("results/02_detHimagScat_{idx}.tif")
    shell:
        """
        python scripts/fun.py propDetVol \
        --input {input.para} {input.scanPara} {input.inPSFreal} {input.inPSFimag} {input.inPropVol}\
        --output {output.outPSFreal} {output.outPSFimag}
        """

rule genSysPSF:
    input:
        para = "data/para.json",
        inPSFrealExc = "results/02_excHrealScat_{idx}.tif",
        inPSFimagExc = "results/02_excHimagScat_{idx}.tif", 
        inPSFrealDet = "results/02_detHrealScat_{idx}.tif",
        inPSFimagDet = "results/02_detHimagScat_{idx}.tif"
    output:
        sysHReal = temp("results/03_sysHReal_{idx}.tif"),
        sysHImag = temp("results/03_sysHImag_{idx}.tif")
    shell:
        """
        python scripts/fun.py genSysPSF \
        --input {input.para} {input.inPSFrealExc} {input.inPSFimagExc} \
                {input.inPSFrealDet} {input.inPSFimagDet}\
        --output {output.sysHReal} {output.sysHImag}
        """

rule genSysPS:
    input:
        para = "data/para.json",
        insysHReal = "results/03_sysHReal_{idx}.tif",
        insysHImag = "results/03_sysHImag_{idx}.tif"
    output:
        outPS = temp("results/03_psSys_{idx}.tif")
    shell:
        """
        python scripts/fun.py genSysPS \
        --input {input.para} {input.insysHReal} {input.insysHImag} \
        --output {output.outPS}
        """

def getGenHistoInputs(wildcards):

    mode = wildcards.mode
    idx = wildcards.idx

    inputs = ["data/para.json", "results/01_thetaVol.tif", "results/01_phiVol.tif"]

    if mode == "sys":
        inputs.append(f"results/03_psSys_{idx}.tif")
    elif mode == "exc":
        inputs.extend([
            f"results/02_excHrealScat_{idx}.tif",
            f"results/02_excHimagScat_{idx}.tif"
        ])
    elif mode == "det":
        inputs.extend([
            f"results/02_detHrealScat_{idx}.tif",
            f"results/02_detHimagScat_{idx}.tif"
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
                   #expand("results/04_resBS_{{mode}}_{idx}.json", 
                   ##mode=wildcards.mode, 
                   #idx=getMaxIDX(wildcards))
    return outs

rule constImag:
    input:
        para = "data/para.json",
        #inScan = "results/01_scanPara.json",
        inScan = "data/para.json",
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