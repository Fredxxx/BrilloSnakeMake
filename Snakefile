<<<<<<< HEAD
p = "/scratch/goerlitz/brilloSM/"

rule all:
    input:
        p + "results/01_propVol.tif",
        p + "results/01_phiVol.tif",
        p + "results/01_thetaVol.tif",
        p + "results/01_excHreal.tif",
        p + "results/01_excHimag.tif",
        p + "results/01_detHreal.tif",
        p + "results/01_detHimag.tif",
        p + "results/02_excHrealScat_0.tif",
        p + "results/02_detHrealScat_0.tif"
=======
import json

def getMaxIDX():
    with open("data/para.json", "r") as f:
        js = json.load(f)
        idxMax = js["scanPara"]["xSteps"]*js["scanPara"]["xSteps"]
    return list(range(idxMax))

with open("data/para.json", "r") as f:
    js = json.load(f)

print(f"config file: {js}")

fields = []
if js["calc"]["sys"] == 1:
    fields.append("sys")
if js["calc"]["exc"] == 1:
    fields.append("exc")
if js["calc"]["det"] == 1:
    fields.append("det")

print(f"fields: {fields}")

modes = []
if js["adv"]["MTF"] ==1:
    modes.append("MTF")
if js["adv"]["OTF"] ==1:
    modes.append("OTF")
if js["adv"]["PS"] ==1:
    modes.append("PS")
if js["adv"]["dodgyI"] ==1:
    modes.append("dodgyI")
if js["adv"]["stat"] ==1:
    modes.append("stat")

print(f"modes: {modes}")

rule all:
    input:
        "results/01_propVol.tif"
        #expand("results/fin_{mode}_{field}", mode = modes, field=fields)
        #expand("results/04_resBS_{mode}_{field}_{idx}.json", mode = modes, field=fields, idx=getMaxIDX())
        #expand("results/03_sysHImag_{mode}_{field}_{idx}.tif", mode = modes, field=fields, idx=getMaxIDX()),
>>>>>>> ed2711b9d0fb88dbd22935a0af784b49735b4750


#rule test:
#    input:
#        para = "data/para.json"
#    output:
#        propVol = "results/01_propVol.tif"
#    threads: 1
#    resources:
#        mem_mb=2000,
#        time_min=2
#        #gpu_mem_mb=24000
#    shell:
#        """
#        python scripts/fun.py test \
#        --input {input.para} \
#        --output {output.propVol} 
#        """

rule loadPadSampleVol:
    input:
        para = "data/para.json"
    output:
<<<<<<< HEAD
        propVol = p + "results/01_propVol.tif"
    threads: 1
    resources:
        slurm_partition="bigmem",
        mem_mb=1024000,
        runtime=60
=======
        propVol = "results/01_propVol.tif"
    threads: 1
    resources:
        mem_mb = 2000
        time_min = 15
        #gpu_mem_mb=24000
>>>>>>> ed2711b9d0fb88dbd22935a0af784b49735b4750
    shell:
        "python scripts/funCPU.py loadPadSampleVol --input {input.para} --output {output}"

rule genExcField:
    input:
        para = "data/para.json" 
    output: 
        excHreal = p + "results/01_excHreal.tif",
        excHimag = p + "results/01_excHimag.tif"
    threads: 1
    resources:
        slurm_partition="gpu-el8",
        constraint="rome,gpu=3090",
        gres="shard:1",
        gpu_mem_mb=64,
        mem_mb=24000,
        runtime=15
    shell:
        """
        python scripts/funGPU.py genExcField \
        --input {input.para}  \
        --output {output.excHreal} {output.excHimag} 
        """
rule genDetField:
    input:
        para = "data/para.json"
    output: 
        detHreal = p + "results/01_detHreal.tif",
        detHimag = p + "results/01_detHimag.tif"
    threads: 1
    resources:
        slurm_partition="gpu-el8",
        constraint="rome,gpu=3090",
        gres="shard:1",
        gpu_mem_mb=64,
        mem_mb=24000,
        runtime=15
    shell:
        """
        python scripts/funGPU.py genDetField \
        --input {input.para} \
        --output {output.detHreal} {output.detHimag} 
        """

rule genAngleSpace:
    input:
        para = "data/para.json"
    output:
        thetaVol = p + "results/01_thetaVol.tif",
        phiVol = p + "results/01_phiVol.tif"
    threads: 1
    resources:
        mem_mb=48000,
        runtime=15
    shell:
        """
        python scripts/funCPU.py genAngleSpace \
        --input {input.para} \
        --output {output.thetaVol} {output.phiVol} 
        """

rule propExcVol:
    input:
        para = "data/para.json",
        inPSFreal = p + "results/01_excHreal.tif",
        inPSFimag = p + "results/01_excHimag.tif",
        inPropVol = p + "results/01_propVol.tif"
    threads: 1
    resources:
        slurm_partition="gpu-el8",
        constraint="rome,gpu=3090",
        gres="shard:1", #"gpu:3090:4"
        gpu_mem_mb=3000,
        mem_mb=76000,
        runtime=30
    output:
        outPSFreal = temp(p + "results/02_excHrealScat_{idx}.tif"),
        outPSFimag = temp(p + "results/02_excHimagScat_{idx}.tif")
    shell:
        """
        python scripts/funGPU.py propExcVol \
        --input {input.para} {input.inPSFreal} {input.inPSFimag} {input.inPropVol}\
        --output {output.outPSFreal} {output.outPSFimag}
        """

rule propDetVol:
    input:
        para = "data/para.json",
        inPSFreal = p + "results/01_detHreal.tif",
        inPSFimag = p + "results/01_detHimag.tif",
        inPropVol = p + "results/01_propVol.tif"
    output:
        outPSFreal = temp(p + "results/02_detHrealScat_{idx}.tif"),
        outPSFimag = temp(p + "results/02_detHimagScat_{idx}.tif")
    threads: 1
    resources:
        slurm_partition="gpu-el8",
        constraint="rome,gpu=3090",
        gres="shard:1", #"gpu:3090:4"
        gpu_mem_mb=3000,
        mem_mb=76000,
        runtime=30
    shell:
        """
        python scripts/funGPU.py propDetVol \
        --input {input.para} {input.inPSFreal} {input.inPSFimag} {input.inPropVol}\
        --output {output.outPSFreal} {output.outPSFimag}
        """