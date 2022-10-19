import os, re
import subprocess

from numpy import size

def open_file(fname):
    with open(fname, "r") as file:
        data = file.read()
    return data

def write_to_file(fname, text):
    f= open(fname, "w")
    f.write(text)


def gen_fakeapp(volume, mode, IOpattern, IOsize, target, nodes=1):
    print("volume: ", volume)
    print("mode: ", mode)
    print("IO Pattern: ", IOpattern)
    print("IO Size: ", IOsize)
    print("Compute nodes: ", nodes)
    print("Backend target: ", target)

    N = int(volume / IOsize)
    print("Number of IO: ", N)

    #read template file
    template = "fakeapp.sbatch"
    sbatch = open_file(template)
    print(sbatch)

    #generate fakeapp from template
    lead = 1
    if (IOpattern=="Seq"):
        lead = 1
    else:
        lead = 3

    #gen_sbatch = re.sub(r"\s$VOLUME", str(volume), sbatch)
    mod_sbatch = sbatch.replace("$VOLUME", str(volume))
    mod_sbatch = mod_sbatch.replace("$OPS", str(N))
    mod_sbatch = mod_sbatch.replace("$LEAD", str(lead))
    mod_sbatch = mod_sbatch.replace("$SIZE", str(IOsize))
    mod_sbatch = mod_sbatch.replace("$NODES", str(nodes))
    mod_sbatch = mod_sbatch.replace("$TARGET", target)
    if mode == "Read":
        mod_sbatch = mod_sbatch.replace("$MODE", "-R")
    else:
        mod_sbatch = mod_sbatch.replace("$MODE", "")

    print("------------------")
    print(mod_sbatch)

    #run generated fakeapp
    write_to_file("mod_sbatch.sbatch", mod_sbatch)
    subprocess.run(["sbatch", "mod_sbatch.sbatch"])

if __name__ == '__main__':
    lfs="/fsiof/phamtt/tmp"
    nfs="/scratch/phamtt/tmp"
    gen_fakeapp(1000000, "Read", "Seq", 1000, nfs)
    gen_fakeapp(1000000, "Write", "Stride", 1000, lfs, 2)