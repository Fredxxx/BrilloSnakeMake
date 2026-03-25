import json

def getScanIDX(wildcards):
    cp = checkpoints.genIDXs.get().output.scanPara
    with open(cp, "r") as f:
        data = json.load(f)
    return expand("results/03_testW_{idx}.txt", idx=data["idxVector"])

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
        "results/03_I_psfErealScat.tif",
        "results/03_I_psfEimagScat.tif",
        "results/03_I_psfDrealScat.tif",
        "results/03_I_psfDimagScat.tif",
        "results/03_II_psSys.tif"
        #getScanIDX

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

#rule testWildcard:
#    input:
#        paraJSON = "results/01_paraTemp.json",
#        scanPara = "results/02_scanPara.json"
#    output:
#        testW = "results/03_testW_{idx}.txt"
#    shell:
#        """
#        python scripts/fun.py testWildcard \
#        --input {input.paraJSON} {input.scanPara}\
#        --output {output.testW}
#        """

rule propExcVol:
    input:
        paraJSON = "results/01_paraTemp.json",
        inPSFreal = "results/02_psfEreal.tif",
        inPSFimag = "results/02_psfEimag.tif" 
    output:
        outPSFreal = temp("results/03_I_psfErealScat.tif"),
        outPSFimag = temp("results/03_I_psfEimagScat.tif")
    shell:
        """
        python scripts/fun.py propExcVol \
        --input {input.paraJSON} {input.inPSFreal} {input.inPSFimag}\
        --output {output.outPSFreal} {output.outPSFimag}
        """

rule propDetVol:
    input:
        inPSFreal = "results/02_psfDreal.tif",
        inPSFimag = "results/02_psfDimag.tif" 
    output:
        outPSFreal = temp("results/03_I_psfDrealScat.tif"),
        outPSFimag = temp("results/03_I_psfDimagScat.tif")
    shell:
        """
        python scripts/fun.py propDetVol \
        --input {input.inPSFreal} {input.inPSFimag}\
        --output {output.outPSFreal} {output.outPSFimag}
        """

rule genPS:
    input:
        inPSFrealExc = "results/03_I_psfErealScat.tif",
        inPSFimagExc = "results/03_I_psfEimagScat.tif", 
        inPSFrealDet = "results/03_I_psfDrealScat.tif",
        inPSFimagDet = "results/03_I_psfDimagScat.tif"
    output:
        outPS = "results/03_II_psSys.tif"
    shell:
        """
        python scripts/fun.py genPS \
        --input {input.inPSFrealExc} {input.inPSFimagExc} \
                {input.inPSFrealDet} {input.inPSFimagDet}\
        --output {output.outPS}
        """