__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""
import os, re
import subprocess
import string, random
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
            accelerator (bool): IO accelerator to be used (sbb), default=False
            ioi (bool): enable/disable the instrumentation using IOI, default = False

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


    @staticmethod
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

        string_time = string_time.replace(f"{minutes}m", "")
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

    def get_slurm_times(self):
        """Parses the slurm times associated with the file slurm-job_id.out

        Args:
            out_file (str): The job slurm output file path to parse.

        Returns:
            The time real value.
        """
        self.output_file = os.path.join(self.working_dir, f"slurm-{str(self.job_id)}.out")
        logger.info(f"Parsing Slurm output file: {self.output_file}")
        real_time = None
        try:
            with open(self.output_file, "r") as slurm_file:
                lines = slurm_file.readlines()
                for line in lines:
                    if line.startswith("real"):
                        time = line.split("real")[-1].strip()
                        real_time = FakeappWorkload.parse_milliseconds(time)
                        logger.info(f"Time found is {time}  | Parsed time is {real_time}")
            if real_time:
                return real_time
            raise ValueError("Slurm command was not timed !")
        except FileNotFoundError as exc:
            raise FileNotFoundError("Slurm output was not found.") from exc


    def write_sbatch_file(self):
        """
        Write down an sbatch file to be used later by slurm. The template is updated using parameters values.

        Args:
            sbatch (str): content of sbatch file
            ioi (bool): enable/disable IOI
            wait (bool): wait until the end of execution (default=True)

        Returns:
            The application elapsed time in seconds
        """
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.working_dir = os.path.join(self.current_dir, "tmp")
        self.sbatch_template = os.path.join(self.current_dir, "defaults", "fakeapp.sbatch")
        lead = 3 if self.io_pattern == "stride" else 1
        scatter = 1000000 if self.io_pattern == "rand" else 0

        # read the content of the template file
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

        # save the generated file to sbatch file
        filename = ''.join(random.choice(string.ascii_letters) for _ in range(7)) + '.sbatch'
        self.sbatch_file = os.path.join(self.current_dir, "tmp", filename)
        with open(self.sbatch_file, "w") as file_temp:
            file_temp.write(template_file)

        logger.info(f"Sbatch file to be used: {self.sbatch_file}")
        logger.trace(f"Sbatch file content: {file_temp}")
        #return self.sbatch_file

    def run_sbatch_file(self, ioi=False, wait=True, clean=False):
        """Generates the command line and run with the appropriate sbatch file.

        Returns:
            ioi (bool): enable/disable IOI
            wait (bool): wait until the end of execution (default=True)
        """
        # If the wait option is enabled, use the --wait flag
        wait_flag = " --wait " if wait else ""

        # If the instrumentation is enabled, instrument the run
        ioi_flag = " --ioi=yes " if ioi else ""

        self.cmd_line = f"sbatch{wait_flag}{ioi_flag}{self.sbatch_file}"
        logger.info(f"Command line to be used: {self.cmd_line}")
        logger.info(f"Working directory: {self.working_dir}")

        # Run the script using subprocess
        sub_ps = subprocess.run(self.cmd_line.split(), cwd=self.working_dir,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output_stdout = sub_ps.stdout.decode()
        output_stderr = sub_ps.stderr.decode()
        # Get and store the job_id from the stdout
        if sub_ps.returncode == 0:
            self.job_id = int(output_stdout.split()[-1])
            logger.info(f"Slurm Job Id: {self.job_id}")

        else:
            logger.error(f"Could not run sbatch file: {self.sbatch_file}")
            raise Exception(f"Could not submit job: \n stderr: {output_stderr}")

        #get avg time from slurm out
        real_time = self.get_slurm_times()
        logger.info(f"Slurm job {self.job_id} is finished | Elapsed time (seconds) : {real_time}")

        # cleaning and renaming
        if clean:
            os.remove(self.sbatch_file)
            os.remove(self.output_file)
        else:
            final_sbatch = os.path.join(self.current_dir, "tmp", f"{self.job_id}.sbatch")
            os.rename(self.sbatch_file, final_sbatch)
            self.sbatch_file = final_sbatch
        return real_time

    def get_data(self):
        bandwidth = 0
        if self.volume > 0:
            self.write_sbatch_file()
            elapsed_time = self.run_sbatch_file(clean=True)
            if elapsed_time > 0:
                bandwidth = self.volume / elapsed_time

        logger.info(f"Workload duration: {elapsed_time} | bandwidth: {convert_size(bandwidth)}/s")
        return elapsed_time, bandwidth













