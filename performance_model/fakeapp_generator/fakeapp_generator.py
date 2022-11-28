__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import os, re
import subprocess
from numpy import size

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

def open_file(fname):
    with open(fname, "r") as file:
        data = file.read()
    return data

def write_to_file(fname, text):
    f= open(fname, "w")
    f.write(text)

def gen_fakeapp(volume, mode, IOpattern, IOsize, nodes, target, accelerator ="", ioi=False):
    """
    Generate an sbatch file using the features extracted for each phase
    by AppDecomposer
    Args:
        volume (int): volume of io will be generated
        mode (string): read/write
        IOpattern (string): Seq/Stride/Random
        IOsize (string): size of IO
        nodes (int): number of nodes, default = 1
        target (string): storage backend file (nfs or lfs)
        accelerator (string): IO accelerator to be used (sbb), default=""
        ioi (bool): enable/disable IOI, default = False

    Returns:
        elap_time (float): elapsed time
        bandwidth (float): IO bandwidth
    """
    print("===================================")
    print("Volume: ", volume)
    print("Mode: ", mode)
    print("IO Pattern: ", IOpattern)
    print("IO Size: ", IOsize)
    print("Compute nodes: ", nodes)
    print("Backend target: ", target)

    N = int(volume / IOsize)
    print("Number of IO: ", N)
    print("IOI Enabled: ", ioi)
    print("IO Accelerator: ", accelerator)

    #read template file
    template = os.path.join(dir_path, "fakeapp.sbatch")
    sbatch = open_file(template)
    #print(sbatch)

    #generate fakeapp from template
    lead = 1
    scatter = 0
    if (IOpattern=="stride"):
        lead = 3
    if (IOpattern=="rand"):
        scatter = 1000000

    mod_sbatch = sbatch.replace("$VOLUME", str(volume))
    mod_sbatch = mod_sbatch.replace("$OPS", str(N))
    mod_sbatch = mod_sbatch.replace("$LEAD", str(lead))
    mod_sbatch = mod_sbatch.replace("$SIZE", str(IOsize))
    mod_sbatch = mod_sbatch.replace("$SCATTER", str(scatter))
    mod_sbatch = mod_sbatch.replace("$NODES", str(nodes))
    mod_sbatch = mod_sbatch.replace("$TARGET", target)
    if mode == "read":
        mod_sbatch = mod_sbatch.replace("$MODE", "-R")
    else:
        mod_sbatch = mod_sbatch.replace("$MODE", "")

    if accelerator =="SBB":
        mod_sbatch = mod_sbatch.replace("$ACC", accelerator)

    print("------------------")

    #run sbacth to get executed time
    elap_time = run_sbatch(mod_sbatch, ioi)

    #compute bandwidth
    bandwidth = get_bandwidth(volume, elap_time)

    return elap_time, bandwidth

def run_sbatch(sbatch, ioi, wait=True):
    """
    Submit sbatch to slurm and get realtime of executing application

    Args:
        sbatch (str): content of sbatch file
        ioi (bool): enable/disable IOI
        wait (bool): wait until the end of execution (default=True)

    Returns:
        The applicaiton elapsed time in seconds
    """
    #write generated script to sbatch file
    sbatch_file=os.path.join(dir_path, "mod_sbatch.sbatch")
    write_to_file(sbatch_file, sbatch)

    # If the wait option is enabled, use the --wait flag
    wait_flag = ""
    if wait:
        wait_flag = " --wait "

    # If the instrumentation is enabled, instrument the run
    ioi_flag = ""
    if ioi:
        ioi_flag = " --ioi=yes "

    cmd_line = "sbatch" + wait_flag + ioi_flag + sbatch_file
    #print("CMD: ", cmd_line)

    # Run the script using subprocess
    sub_ps = subprocess.run(cmd_line.split(), cwd=dir_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output_stdout = sub_ps.stdout.decode()
    output_stderr = sub_ps.stderr.decode()
    # Get and store the job_id from the stdout
    if sub_ps.returncode == 0:
        job_id = int(output_stdout.split()[-1])
        print("Job id:", job_id)
    else:
        print("Could not run sbatch file:", sbatch_file)
        raise Exception(f"Could not submit job: \n stderr: {output_stderr}")

    #get avg time from slurm out
    real_time = get_slurm_times(job_id)
    print("Elapsed time (seconds) : ", real_time)

    return real_time

def get_slurm_times(job_id):
    """Parses the slurm times associated with the file slurm-job_id.out

    Args:
        out_file (str): The job slurm output file path to parse.

    Returns:
        The time real value
    """
    out_file = os.path.join(dir_path, "slurm-" + str(job_id) + ".out")
    print("Slurm output: ", out_file)
    real_time = None
    try:
        with open(out_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("real"):
                    time = line.split("real")[-1].strip()
                    real_time = parse_milliseconds(time)
        if real_time:
            return real_time
        raise ValueError(f"Slurm command was not timed !")
    except FileNotFoundError as exc:
        raise FileNotFoundError("Slurm output was not found.") from exc

def parse_milliseconds(string_time):
    """
    Given a string date with the format MMmSS.MSs (as returned by the linux time command),
    parses it to seconds.

    Args:
        string_time (str): The date to parse

    Returns:
        The number of elapsed seconds
    """
    minutes = 0
    seconds = 0
    milliseconds = 0
    try:
        minutes = int(string_time.split("m")[0])
    except ValueError:
        pass

    string_time = string_time.replace(str(minutes) + "m", "")
    try:
        seconds = int(string_time.split(".")[0])
    except ValueError:
        pass

    milliseconds_string = string_time.split(".")[1]
    milliseconds_string = milliseconds_string.replace("s", "")
    try:
        milliseconds = int(milliseconds_string)
    except ValueError:
        pass

    return minutes * 60 + seconds + milliseconds / 1000

def get_bandwidth(volume, time):
    """
    Compute the IO bandwidth realized by the application

    Args:
        volume (int): IO volume
        time (float): elapsed time

    Returns:
        IO bandwidth in Mb/s
    """
    #compute IO bandwidth
    bw = 0
    if (time > 0):
        bw = volume/time

    print("IO bandwidth: ", format(bw/(1024*1024), '.2f'), "(Mb/s)")

    return bw

if __name__ == '__main__':
    lfs="/fsiof/phamtt/tmp"
    nfs="/scratch/phamtt/tmp"
    acc = "SBB"
#    gen_fakeapp(1000000000, "read", "seq", 10000, 1, nfs,"", True)
    gen_fakeapp(1000000000, "write", "rand", 10000, 1, lfs,"", False)
    #gen_fakeapp(1000000000, "Write", "Seq", 1000, nfs, 2, True)
    #gen_fakeapp(10000000, "Read", "Stride", 1000, lfs, 2, True)
    #gen_fakeapp(10000000, "Read", "Seq", 1000, lfs, 2, True)