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


def gen_fakeapp(volume, mode, IOpattern, IOsize, target, nodes=1, ioi=False, accelerator =""):
    """
    Generate an sbatch file using the features extracted for each phase
    by AppDecomposer
    Args:
        volume (int): volume of io will be generated
        mode (string): read/write
        IOpattern (string): Seq/Stride/Random
        IOsize (string): size of IO
        target (string): storage backend file (nfs/fs1/sbb)
        nodes (int): number of nodes, default = 1
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
    template = "fakeapp.sbatch"
    sbatch = open_file(template)
    #print(sbatch)

    #generate fakeapp from template
    lead = 1
    scatter = 0
    if (IOpattern=="Stride"):
        lead = 3
    if (IOpattern=="Random"):
        scatter = 1000000

    #gen_sbatch = re.sub(r"\s$VOLUME", str(volume), sbatch)
    mod_sbatch = sbatch.replace("$VOLUME", str(volume))
    mod_sbatch = mod_sbatch.replace("$OPS", str(N))
    mod_sbatch = mod_sbatch.replace("$LEAD", str(lead))
    mod_sbatch = mod_sbatch.replace("$SIZE", str(IOsize))
    mod_sbatch = mod_sbatch.replace("$SCATTER", str(scatter))
    mod_sbatch = mod_sbatch.replace("$NODES", str(nodes))
    mod_sbatch = mod_sbatch.replace("$TARGET", target)
    if mode == "Read":
        mod_sbatch = mod_sbatch.replace("$MODE", "-R")
    else:
        mod_sbatch = mod_sbatch.replace("$MODE", "")

    if accelerator =="SBB":
        mod_sbatch = mod_sbatch.replace("$ACC", accelerator)
    else:
        mod_sbatch = mod_sbatch.replace("$ACC", "no")

    print("------------------")

    #run sbacth to get executed time
    elap_time = run_sbatch(mod_sbatch, ioi)

    #compute bandwidth
    bandwidth = get_bandwidth(volume, elap_time)

    return elap_time, bandwidth

def run_sbatch(sbatch, ioi, wait=True):
    #write generated script to sbatch file
    sbatch_file="mod_sbatch.sbatch"
    write_to_file(sbatch_file, sbatch)

    # If the wait option is enabled, use the --wait flag
    wait_flag = ""
    if wait:
        wait_flag = " --wait "

    # If the instrumentation is enabled, instrument the run
    ioi_flag = ""
    if ioi:
        ioi_flag = " --ioi=yes "

    cmd_line = "sbatch" + wait_flag + ioi_flag + "mod_sbatch.sbatch"

    # Run the script using subprocess
    sub_ps = subprocess.run(cmd_line.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output_stdout = sub_ps.stdout.decode()
    output_stderr = sub_ps.stderr.decode()
    # Get and store the job_id from the stdout
    if sub_ps.returncode == 0:
        job_id = int(output_stdout.split()[-1])
        print("Job id:", job_id)
    else:
        print("Could not run sbatch:", output_stderr)
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
    cwd = os.getcwd() #get current directory
    out_file = os.path.join(cwd, "slurm-" + str(job_id) + ".out")
    print("Slurm output: ", out_file)
    real_time = None
    try:
        with open(out_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("real"):
                    time = re.split('[ \t]*', line)[-1].strip()
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
    minutes = int(string_time.split("m")[0])
    string_time = string_time.replace(str(minutes) + "m", "")
    seconds = int(string_time.split(".")[0])
    milliseconds_string = string_time.split(".")[1]
    milliseconds_string = milliseconds_string.replace("s", "")
    milliseconds = int(milliseconds_string)
    return minutes * 60 + seconds + milliseconds / 1000

def get_bandwidth(volume, time):

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
    gen_fakeapp(1000000000, "Write", "Random", 10000, lfs, 2, True, acc)
    gen_fakeapp(1000000000, "Write", "Random", 10000, lfs, 2, True)
    #gen_fakeapp(1000000000, "Write", "Seq", 1000, nfs, 2, True)
    #gen_fakeapp(10000000, "Read", "Stride", 1000, lfs, 2, True)
    #gen_fakeapp(10000000, "Read", "Seq", 1000, lfs, 2, True)