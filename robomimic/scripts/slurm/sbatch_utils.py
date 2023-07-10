"""
Python script for generating and executing sbatch files
"""

import os
# import slurm
import robomimic
from pathlib import Path

PARTITIONS = (
    "napoli",
    "tibet",
    "svl",

    "titans",
    "dgx",
)

robomimic_base_path = os.path.abspath(os.path.join(os.path.dirname(robomimic.__file__), os.pardir))

AUTO_OVERWRITE_RESP = os.path.join(robomimic_base_path, "robomimic/scripts/slurm/auto_overwrite.txt")
AUTO_APPEND_RESP = os.path.join(robomimic_base_path, "robomimic/scripts/slurm/auto_append.txt")

import time
from datetime import datetime

def create_and_execute_sbatch_script(filename, job_name, sbatch_args, script_args=None):
    """
    Function that creates and executes an sbatch script based off of a template

    Args:
        @filename (str): Name of the sbatch file that will be generated
        @job_name (str): Name of sbatch job to execute
        @sbatch_args (Namespace): Input arguments to fill in sbatch script
        @script_args (list of dicts, dict or None): If specified, adds additional
            input arguments to script execution based on key-value mappings.
            If of type list, indicates multiple commands in one sbatch script.
    """
    # Create a new directory path if it doesn't exist and create a new filename that we will write to
    Path(sbatch_args.generated_dir).mkdir(parents=True, exist_ok=True)
    ts = time.time()
    new_sbatch_fpath = os.path.join(sbatch_args.generated_dir, "{}_{}.sbatch".format(filename, ts))

    # Compose extra commands
    if sbatch_args.extra_commands is not None:
        sbatch_args.extra_commands = sbatch_args.extra_commands if type(sbatch_args.extra_commands) is list else \
            [sbatch_args.extra_commands]
        sbatch_args.extra_commands = "\n".join(sbatch_args.extra_commands)
    else:
        sbatch_args.extra_commands = ""

    # infer number of commands from script args
    if script_args is None:
        num_commands = 1
    elif not isinstance(script_args, list):
        script_args = [script_args]
        num_commands = 1
    else:
        num_commands = len(script_args)

    command = ""
    for i in range(num_commands):
        # Compose main command to be executed in script
        command += "python {}".format(sbatch_args.script)

        # Add additional input args if necessary
        if script_args is not None:
            for k, v in script_args[i].items():
                if v is not None:
                    if type(v) is list or type(v) is tuple:
                        v = " ".join(str(vi) for vi in v)
                    command += " --{} {}".format(k, v)

        # Add overwrite if requested
        if sbatch_args.overwrite:
            command += f" < {AUTO_OVERWRITE_RESP}"
        else:
            command += f" < {AUTO_APPEND_RESP}"

        command += " & \n"
    command += "wait"

    # Define partition
    if sbatch_args.partition == "napoli":
        partition = "napoli-gpu" if sbatch_args.num_gpu > 0 else "napoli-cpu\n#SBATCH --exclude=napoli[15-16]"
    else:
        partition = sbatch_args.partition

    # Define GPU(s) to use
    num_gpu = sbatch_args.num_gpu
    if sbatch_args.gpu_type != "any":
        num_gpu = f"{sbatch_args.gpu_type}:{num_gpu}"

    # Add copy file if requested
    copy_file = "" if sbatch_args.copy_file is None else create_copy_file_cmd(*sbatch_args.copy_file)

    # Add shell source script if requested
    shell_source_script = "" if sbatch_args.shell_source_script is None else f"source {sbatch_args.shell_source_script}"

    # Define a dict to map expected fill-ins with replacement values
    fill_ins = {
        "{{PARTITION}}": partition,
        "{{EXCLUDE}}": sbatch_args.exclude,
        "{{NUM_GPU}}": num_gpu,
        "{{NUM_CPU}}": sbatch_args.num_cpu,
        "{{JOB_NAME}}": job_name,
        "{{EXECUTABLE_LOG_DIR}}": sbatch_args.executable_log_dir,
        "{{HOURS}}": sbatch_args.max_hours,
        "{{QOS_LONG}}": "#SBATCH --qos=long" if sbatch_args.max_hours > 48 else "",
        "{{MEM}}": sbatch_args.mem_gb,
        "{{NOTIFICATION_EMAIL}}": sbatch_args.notification_email,
        "{{SHELL_SOURCE_SCRIPT}}": shell_source_script,
        "{{PYTHON_INTERPRETER}}": sbatch_args.python_interpreter,
        "{{EXTRA_PYTHONPATH}}": sbatch_args.extra_pythonpath,
        "{{MUJOCO_DIR}}": sbatch_args.mujoco_dir,
        "{{COPY_FILE}}": copy_file,
        "{{CMD}}": command,
        "{{EXTRA_CMDS}}": sbatch_args.extra_commands
    }

    # Open the template file
    with open(os.path.join(robomimic_base_path, "robomimic/scripts/slurm/base_template.sbatch")) as template:
        # Open the new sbatch file
        print(new_sbatch_fpath)
        with open(new_sbatch_fpath, 'w+') as new_file:
            # Loop through template and write to this new file
            for line in template:
                wrote = False
                # Check for various cases
                for k, v in fill_ins.items():
                    # If the key is found in the line, replace it with its value and pop it from the dict
                    if k in line:
                        new_file.write(line.replace(k, str(v)))
                        wrote = True
                        break
                # Otherwise, we just write the line from the template directly
                if not wrote:
                    new_file.write(line)

    # Execute this file!
    # TODO: Fix! (Permission denied error)
    #os.system(new_sbatch_fpath)


def create_copy_file_cmd(source_file, target_dir):
    """
    Helper function to create a bash command (in string format) to copy a source file to a target location.

    Args:
        source_file (str): Absolute path to the source file to copy
        target_dir (str): Absolute path to the target directory to which the source file will be copied

    Returns:
        str: bash command to execute in string format
    """
    target_filename = source_file.split("/")[-1]
    target_fpath = os.path.join(target_dir, target_filename)
    cmd =\
        f'mkdir -p {target_dir}\n'\
        f'if [[ -f "{target_fpath}" ]]; then\n'\
        f'    echo "{target_fpath} exists, no copying"\n'\
        f'else\n'\
        f'    echo "{target_fpath} does not exist, copying dataset"\n'\
        f'    cp {source_file} {target_fpath}\n'\
        f'fi'

    return cmd
