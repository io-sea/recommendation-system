__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""
import os, re
import subprocess
from numpy import size
from loguru import logger
from app_decomposer.utils import convert_size

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

# def open_file(fname):
#     with open(fname, "r") as file:
#         data = file.read()
#     return data

# def write_to_file(fname, text):
#     f= open(fname, "w")
#     f.write(text)

class FakeappWorkload:
    """Class to create and manage an IO Workload using fakeapp binary. """
    def __init__(self, volume=1e9, mode="read", io_pattern="rand", io_size=4e3, nodes=1,
                 target_tier="lfs", accelerator=False, ioi=False):
        """
        Generate an sbatch file using the features extracted for each phase
        by AppDecomposer
        Args:
            volume (int): volume of io that will be generated
            mode (string): read/write
            io_pattern (string): Seq/Stride/Random
            io_size (string): size of IO
            nodes (int): number of nodes, default = 1
            target (string): storage backend file (nfs or lfs)
            accelerator (bool): IO accelerator to be used (sbb), default=""
            ioi (bool): enable/disable IOI, default = False

        Returns:
            elapsed_time (float): elapsed time
            bandwidth (float): IO bandwidth
        """
        self.volume = volume
        self.mode = mode
        self.io_pattern = io_pattern
        self.io_size = io_size
        self.nodes = nodes
        self.target_tier = target_tier
        self.accelerator = accelerator
        self.ioi = ioi
        self.ios = int(self.volume / self.io_size) if self.io_size else 0
        # init logging
        logger.info(f"Volume: {convert_size(self.volume)} | Mode: {self.mode} | IO pattern: {self.io_pattern} | IO size: {convert_size(self.io_size)} | Nodes: {self.nodes} | Storage tier: {self.target_tier}")
        logger.info(f"#IO: {self.ios} | IOI enabled: {self.ioi} | SBB Accelerated: {self.accelerator}")

    def updated_sbatch_template(self):
        """Using workload attributes to update accordingly the sbatch template file."""
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.sbatch_template = os.path.join(self.current_dir, "defaults", "fakeapp.sbatch")
        lead = 3 if self.io_pattern == "stride" else 1
        scatter = 1000000 if self.io_pattern == "rand" else 0

        # open file for modification
        with open(self.sbatch_template, "r") as temp_file:
            template_file = temp_file.read()

        for entry, value in {"$VOLUME": str(self.volume),
                             "$OPS": str(self.ios),
                             "$LEAD": str(lead),
                             "$SIZE": str(self.io_size),
                             "$SCATTER": str(scatter),
                             "$NODES": str(self.nodes),
                             "$TARGET": self.target_tier}.items():
            template_file = template_file.replace(entry, value)
        if self.mode == "read":
            template_file = template_file.replace("$MODE", "-R")
        else:
            template_file = template_file.replace("$MODE", "")
        if self.accelerator:
            template_file = template_file.replace("$ACC", "SBB")

        # save the modified sbatch file
        return template_file


    def run_sbatch(self, sbatch, ioi=False, wait=True):
        """
        Submit sbatch to slurm and get realtime of executing application

        Args:
            sbatch (str): content of sbatch file
            ioi (bool): enable/disable IOI
            wait (bool): wait until the end of execution (default=True)

        Returns:
            The application elapsed time in seconds
        """
        #write generated script to sbatch file
        sbatch_file = os.path.join(self.current_dir, "defaults", "mod_sbatch.sbatch")
        with open(sbatch_file, "w") as file_temp:
            file_temp.write(sbatch)


        # If the wait option is enabled, use the --wait flag
        wait_flag = ""
        if wait:
            wait_flag = " --wait "

        # If the instrumentation is enabled, instrument the run
        ioi_flag = ""
        if ioi:
            ioi_flag = " --ioi=yes "

        cmd_line = f"sbatch{wait_flag}{ioi_flag}{sbatch_file}"
        return cmd_line
        #print("CMD: ", cmd_line)

    #     # Run the script using subprocess
    #     sub_ps = subprocess.run(cmd_line.split(), cwd=dir_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     output_stdout = sub_ps.stdout.decode()
    #     output_stderr = sub_ps.stderr.decode()
    #     # Get and store the job_id from the stdout
    #     if sub_ps.returncode == 0:
    #         job_id = int(output_stdout.split()[-1])
    #         print("Job id:", job_id)
    #     else:
    #         print("Could not run sbatch file:", sbatch_file)
    #         raise Exception(f"Could not submit job: \n stderr: {output_stderr}")

    #     #get avg time from slurm out
    #     real_time = get_slurm_times(job_id)
    #     print("Elapsed time (seconds) : ", real_time)

    #     return real_time











