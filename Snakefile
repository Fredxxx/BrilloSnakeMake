import json

def getScanIDX(wildcards):
    cp = checkpoints.genIDXs.get().output.scanPara
    with open(cp, "r") as f:
        data = json.load(f)
    return data["idxVector"]

def getGenHistoInputs(wildcards):
    """Return inputs for genHisto depending on mode and idx."""
    with open("results/01_paraTemp.json") as f:
        js = json.load(f)

    mode = wildcards.mode
    idx = wildcards.idx

    # Prüfen, ob der Mode überhaupt aktiviert ist
    if (mode == "sys" and js["calc"]["sys"] != 1) or \
       (mode == "exc" and js["calc"]["exc"] != 1) or \
       (mode == "det" and js["calc"]["det"] != 1):
        # Kein Input → Mode ist inaktiv
        return {}

    # Grundinputs
    inputs = {"paraJSON": "results/01_paraTemp.json"}

    # Optionale Inputs abhängig vom Mode
    if mode == "sys":
        inputs["inPS"] = f"results/03_II_psSys_{idx}.tif"
    elif mode == "exc":
        inputs["inPSFreal"] = f"results/03_I_psfErealScat_{idx}.tif"
        inputs["inPSFimag"] = f"results/03_I_psfEimagScat_{idx}.tif"
    elif mode == "det":
        inputs["inPSFrealDet"] = f"results/03_I_psfDrealScat_{idx}.tif"
        inputs["inPSFimagDet"] = f"results/03_I_psfDimagScat_{idx}.tif"

    return inputs

with open("results/01_paraTemp.json") as f:
    js = json.load(f)

modes = []
if js["calc"]["sys"] == 1:
    modes.append("sys")
if js["calc"]["exc"] == 1:
    modes.append("exc")
if js["calc"]["det"] == 1:
    modes.append("det")

rule all:
    input:
        "results/01_paraTemp.json",
        "results/02_propVol.tif",
        "results/02_psfEreal.tif",
        "results/02_psfEimag.tif",
        "results/02_psfDreal.tif",
        "results/02_psfDimag.tif",
        "results/02_thetaVol.tif",
        "results/02_phiVol.tif",
        "results/02_scanPara.json",
        expand("results/03_I_psfErealScat_{idx}.tif", idx=getScanIDX),
        expand("results/03_I_psfEimagScat_{idx}.tif", idx=getScanIDX),
        expand("results/03_I_psfDrealScat_{idx}.tif", idx=getScanIDX),
        expand("results/03_I_psfDimagScat_{idx}.tif", idx=getScanIDX),
        expand("results/03_II_psfSysReal_{idx}.tif", idx=getScanIDX),
        expand("results/03_II_psfSysImag_{idx}.tif", idx=getScanIDX),
        expand("results/03_II_psSys_{idx}.tif", idx=getScanIDX),
        expand("results/03_III_resAngles_{mode}_{idx}.json", mode=["sys","exc","det"], idx=getScanIDX)

rule loadPara:
    input:
        paraTXT = "data/para.txt"
    output:
        paraJSON = "results/01_paraTemp.json"
    shell:
        """
        python scripts/fun.py loadPara \
        --input {input.paraTXT} \
        --output {output.paraJSON}
        """

rule loadPadSampleVol:
    input:
        paraJSON = "results/01_paraTemp.json"
    output:
        propVol = "results/02_propVol.tif"
    shell:
        """
        python scripts/fun.py loadPadSampleVol \
        --input {input.paraJSON} \
        --output {output.propVol}
        """

rule genExcPSF:
    input:
        paraJSON = "results/01_paraTemp.json" 
    output: 
        psfEreal = "results/02_psfEreal.tif",
        psfEimag = "results/02_psfEimag.tif"
    shell:
        """
        python scripts/fun.py genExcPSF \
        --input {input.paraJSON}  \
        --output {output.psfEreal} {output.psfEimag} 
        """

rule genDetPSF:
    input:
        paraJSON = "results/01_paraTemp.json" 
    output: 
        psfDreal = "results/02_psfDreal.tif",
        psfDimag = "results/02_psfDimag.tif"
    shell:
        """
        python scripts/fun.py genDetPSF \
        --input {input.paraJSON} \
        --output {output.psfDreal} {output.psfDimag} 
        """

rule genAngleSpace:
    input:
        paraJSON = "results/01_paraTemp.json" 
    output:
        thetaVol = "results/02_thetaVol.tif",
        phiVol = "results/02_phiVol.tif"
    shell:
        """
        python scripts/fun.py genAngleSpace \
        --input {input.paraJSON} \
        --output {output.thetaVol} {output.phiVol} 
        """

checkpoint genIDXs:
    input:
        paraJSON = "results/01_paraTemp.json" 
    output:
        scanPara = "results/02_scanPara.json"
    shell:
        """
        python scripts/fun.py genIDXs \
        --input {input.paraJSON} \
        --output {output.scanPara}
        """

rule propExcVol:
    input:
        paraJSON = "results/01_paraTemp.json",
        scanPara = "results/02_scanPara.json",
        inPSFreal = "results/02_psfEreal.tif",
        inPSFimag = "results/02_psfEimag.tif",
        inPropVol = "results/02_propVol.tif"
    output:
        outPSFreal = temp("results/03_I_psfErealScat_{idx}.tif"),
        outPSFimag = temp("results/03_I_psfEimagScat_{idx}.tif")
    shell:
        """
        python scripts/fun.py propExcVol \
        --input {input.paraJSON} {input.scanPara} {input.inPSFreal} {input.inPSFimag} {input.inPropVol}\
        --output {output.outPSFreal} {output.outPSFimag}
        """

rule propDetVol:
    input:
        paraJSON = "results/01_paraTemp.json",
        scanPara = "results/02_scanPara.json",
        inPSFreal = "results/02_psfDreal.tif",
        inPSFimag = "results/02_psfDimag.tif",
        inPropVol = "results/02_propVol.tif"
    output:
        outPSFreal = temp("results/03_I_psfDrealScat_{idx}.tif"),
        outPSFimag = temp("results/03_I_psfDimagScat_{idx}.tif")
    shell:
        """
        python scripts/fun.py propDetVol \
        --input {input.paraJSON} {input.scanPara} {input.inPSFreal} {input.inPSFimag} {input.inPropVol}\
        --output {output.outPSFreal} {output.outPSFimag}
        """

rule genSysPSF:
    input:
        paraJSON = "results/01_paraTemp.json",
        inPSFrealExc = "results/03_I_psfErealScat_{idx}.tif",
        inPSFimagExc = "results/03_I_psfEimagScat_{idx}.tif", 
        inPSFrealDet = "results/03_I_psfDrealScat_{idx}.tif",
        inPSFimagDet = "results/03_I_psfDimagScat_{idx}.tif"
    output:
        psfSysReal = temp("results/03_II_psfSysReal_{idx}.tif"),
        psfSysImag = temp("results/03_II_psfSysImag_{idx}.tif")
    shell:
        """
        python scripts/fun.py genSysPSF \
        --input {input.paraJSON} {input.inPSFrealExc} {input.inPSFimagExc} \
                {input.inPSFrealDet} {input.inPSFimagDet}\
        --output {output.psfSysReal} {output.psfSysImag}
        """

rule genSysPS:
    input:
        paraJSON = "results/01_paraTemp.json",
        inPsfSysReal = "results/03_II_psfSysReal_{idx}.tif",
        inPsfSysImag = "results/03_II_psfSysImag_{idx}.tif"
    output:
        outPS = temp("results/03_II_psSys_{idx}.tif")
    shell:
        """
        python scripts/fun.py genSysPS \
        --input {input.paraJSON} {input.inPsfSysReal} {input.inPsfSysImag} \
        --output {output.outPS}
        """

rule genHisto:
    input:
        getGenHistoInputs
    output:
        outRes = temp("results/03_III_resAngles_{mode}_{idx}.json")
    shell:
        """
        python scripts/fun.py genHisto \
            --input {input.paraJSON} \
            {input.get('inPS','')} {input.get('inPSFreal','')} {input.get('inPSFimag','')} \
            {input.get('inPSFrealDet','')} {input.get('inPSFimagDet','')} \
            --output {output.outRes}
        """