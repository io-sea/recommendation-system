__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""
import os, re
import time
import subprocess
import string, random
from numpy import size
from loguru import logger
from app_decomposer.utils import convert_size

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)


MAX_RETRIES = 10
RETRY_DELAY = 10  # time to wait between retries, in seconds

class FakeappWorkload:
    """Class to create and manage an IO Workload using fakeapp binary. """
    def __init__(self, phase, target_tier="lfs", accelerator=False, ioi=False):
        """
        Generate an sbatch file using the features extracted for each phase
        by AppDecomposer
        Args:
            phase (dict): IO features extracted by appdecomposer
            target (string): storage backend file (nfs or lfs)
            accelerator (bool): IO accelerator to be used (sbb), default=False
            ioi (bool): enable/disable the instrumentation using IOI, default = False
        """
        self.phase = phase
        self.target_tier = target_tier
        self.accelerator = accelerator
        self.ioi = ioi
        self.phase["read_ops"] = int(self.phase["read_volume"] / self.phase["read_io_size"]) if self.phase["read_io_size"] else 0
        self.phase["write_ops"] = int(self.phase["write_volume"] / self.phase["write_io_size"]) if self.phase["write_io_size"] else 0
        # init logging
        logger.info(f'Volume read: {convert_size(self.phase["read_volume"])} | IO pattern: {self.phase["read_io_pattern"]} | IO size: {self.phase["read_io_size"]} | IO nbr: {self.phase["read_ops"]}')
        logger.info(f'Volume write: {convert_size(self.phase["write_volume"])} | IO pattern: {self.phase["write_io_pattern"]} | IO size: {self.phase["write_io_size"]} | IO nbr: {self.phase["write_ops"]}')
        logger.info(f'Nodes: {self.phase["nodes"]} | Storage tier: {self.target_tier} | SBB Accelerated: {self.accelerator} | IOI enabled: {self.ioi}')


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
        """Parses the slurm times associated with the file slurm-job_id.out.

        Args:
            out_file (str): The job slurm output file path to parse.

        Returns:
            The time real value.
        """
        self.output_file = os.path.join(self.working_dir, f"slurm-{str(self.job_id)}.out")
        logger.info(f"Parsing Slurm output file: {self.output_file}")

        retry_count = 0
        while retry_count < MAX_RETRIES:
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
            except FileNotFoundError:
                logger.warning(f"Slurm output file {self.output_file} not found, retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
                retry_count += 1

        # if the file still not found after all retries
        raise FileNotFoundError(f"Slurm output file {self.output_file} not found after {MAX_RETRIES} retries.")


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

        #compute lead and scatter for stride a and random pattern
        lead_r = 3 if self.phase["read_io_pattern"] == "stride" else 1
        lead_w = 3 if self.phase["write_io_pattern"] == "stride" else 1
        scatter_r = 1e9 if self.phase["read_io_pattern"] == "rand" else 0
        scatter_w = 1e9 if self.phase["write_io_pattern"] == "rand" else 0

        # read the content of the template file
        with open(self.sbatch_template, "r") as temp_file:
            template_file = temp_file.read()

        for entry, value in {"$OPS_R": str(self.phase["read_ops"]),
                             "$OPS_W": str(self.phase["write_ops"]),
                             "$LEAD_R": str(lead_r),
                             "$LEAD_W": str(lead_w),
                             "$SIZE_R": str(self.phase["read_io_size"]),
                             "$SIZE_W": str(self.phase["write_io_size"]),
                             "$SCATTER_R": str(scatter_r),
                             "$SCATTER_W": str(scatter_w),
                             "$NODES": str(self.phase["nodes"]),
                             "$TARGET": self.target_tier}.items():
            template_file = template_file.replace(entry, value)

        #set up accelerator
        if self.accelerator:
            template_file = template_file.replace("$ACC", "SBB")

        # save the generated file to sbatch file
        filename = ''.join(random.choice(string.ascii_letters) for _ in range(7)) + '.sbatch'
        self.sbatch_file = os.path.join(self.current_dir, "tmp", filename)
        with open(self.sbatch_file, "w") as file_temp:
            file_temp.write(template_file)

        logger.info(f"Sbatch file written to: {self.sbatch_file}")
        logger.trace(f"Sbatch file content: {file_temp}")

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
        logger.info(f"Command line run the sbatch file: {self.cmd_line}")
        logger.info(f"Working directory for the Slurm Job: {self.working_dir}")

        # Run the script using subprocess
        sub_ps = subprocess.run(self.cmd_line.split(), cwd=self.working_dir,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output_stdout = sub_ps.stdout.decode()
        output_stderr = sub_ps.stderr.decode()
        # Get and store the job_id from the stdout
        if sub_ps.returncode == 0:
            self.job_id = int(output_stdout.split()[-1])
            logger.info(f"Scheduled slurm job with id: {self.job_id}")

        else:
            logger.error(f"Could not run sbatch file: {self.sbatch_file}")
            raise Exception(f"Could not submit job: \n stderr: {output_stderr}")
        
        # Add this block to check the job status and wait for it to finish
        if not wait:
            while True:
                time.sleep(5)  # wait for 5 seconds before checking the job status again
                check_job = f"squeue -j {self.job_id}"
                check_ps = subprocess.run(check_job.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                check_stdout = check_ps.stdout.decode()
                if str(self.job_id) not in check_stdout:  # if the job_id is not in the output, the job has finished
                    break

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
        """
        Calculate elapsed time and bandwidth for a workload.

        This method calculates the elapsed time and bandwidth for a workload by adding the read and write volume and then dividing the total volume by the elapsed time. The write_sbatch_file method is called to write a batch file, which is then run using the run_sbatch_file method. The elapsed time is obtained from the run_sbatch_file method and used to calculate the bandwidth. The method logs the calculated values and returns them.

        Returns:
        Tuple[float, float]: A tuple of the elapsed time and bandwidth for the workload.
        """
        bandwidth = 0
        elapsed_time = 0
        total_volume = self.phase["read_volume"] + self.phase["write_volume"]
        if  total_volume > 0:
            self.write_sbatch_file()
            elapsed_time = self.run_sbatch_file(self.ioi, clean=True)
            if elapsed_time > 0:
                bandwidth = total_volume / elapsed_time

        logger.info(f"Workload duration: {elapsed_time} | bandwidth: {convert_size(bandwidth)}/s")
        return elapsed_time, bandwidth


# if __name__ == '__main__':
#     lfs="/fsiof/mimounis/tmp"
#     nfs="/scratch/mimounis/tmp"
#     acc = "SBB" # currently support onyly SBB with the lfs target

#     phase0=dict(read_volume=100000000, read_io_pattern="stride", read_io_size=10000, write_volume=0, write_io_pattern="uncl", write_io_size=0, nodes=1)
#     phase0=dict(read_volume=5e7, read_io_pattern="stride", read_io_size=10000,
#                 write_volume=5e7, write_io_pattern="rand", write_io_size=10000, nodes=1)

#     fa = FakeappWorkload(phase0, target_tier=lfs, accelerator=True, ioi=False)
#     fa.get_data()

