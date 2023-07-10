"""
Script for executing all configs generated from hyperparamter_helper.py (in batchRL)

Note that this assumes that hyperparameter_helper.py has already been run, and that all the resulting
configurations exist in a single folder
"""

# from slurm.util.arguments import *
from robomimic.scripts.slurm.batchrl_args import *
from robomimic.scripts.slurm.sbatch_args import *

# from slurm.util.sbatch_utils import create_and_execute_sbatch_script
from robomimic.scripts.slurm.sbatch_utils import create_and_execute_sbatch_script

import copy

# Add relevant input arguments
add_sbatch_args()
add_batchrl_hp_args()


def parse_configs_from_hp_script(hp_script):
    """
    Helper script to parse the executable hyperparameter script generated from hyperparameter_helper.py (in batchRL)
    to infer the filepaths to the generated configs.

    Args:
        hp_script (str): Absolute fpath to the generated hyperparameter script

    Returns:
        list: Absolute paths to the configs to be deployed in the hp sweep
    """
    # Create list to fill as we parse the script
    configs = []
    # Open and parse file line by line
    with open(hp_script) as f:
        for line in f:
            # Make sure we only parse the lines where we have a valid python command
            if line.startswith("python"):
                # Extract only the config path
                configs.append(line.split(" ")[-1].split("\n")[0])
    # Return configs
    return configs


def generate_debug_script(hp_script):
    """
    Helper script to generate an .sh executable debug hyperparameter script using the hp sweep script generated from
    hyperparameter_helper.py (in batchRL)

    Args:
        hp_script (str): Absolute fpath to the generated hyperparameter script
    """
    # Modify the path so that we add "_debug" to the end -- hacky way since we know ".sh" extension is 3 chars long
    debug_script = hp_script[:-3] + "_debug.sh"
    # Open and parse file line by line
    with open(hp_script) as f:
        # Open a new file to write the debug script to
        with open(debug_script, 'w+') as new_file:
            # Loop through hp script and write to this new file
            for line in f:
                # Make sure we only parse the lines where we have a valid python command
                if line.startswith("python"):
                    # We write the line plus the extra --debug flag
                    new_file.write(line.split("\n")[0] + " --debug\n")
                else:
                    # Just write line normally
                    new_file.write(line)


if __name__ == '__main__':
    # First, parse args
    args = parser.parse_args()

    # Extract configs from hp sweep script
    configs = parse_configs_from_hp_script(hp_script=args.batchrl_hp.hp_sweep_script)

    # If user requested a debug script to be generated, do that now
    if args.batchrl_hp.generate_debug_script:
        generate_debug_script(hp_script=args.batchrl_hp.hp_sweep_script)

    n = args.batchrl_hp.n_exps_per_instance

    # Loop through each config to create an sbatch script from
    for i in range(0, len(configs), n):
        script_args = []
        configs_for_batch = configs[i:i+n]
        for config in configs_for_batch:
            # Extract name for this sbatch script
            name = config.split("/")[-1].split(".json")[0]

            # Compose script arguments to pass to sbatch script
            script_args.append({
                "config": config,
            })

            # Generate the sbatch file
            print(f"Creating {name}...")

        # Multiple resources by number of jobs in batch
        sbatch_args = copy.deepcopy(args.sbatch)
        sbatch_args.num_cpu *= len(configs_for_batch)
        sbatch_args.mem_gb *= len(configs_for_batch)

        create_and_execute_sbatch_script(
            filename=name,
            job_name=name,
            sbatch_args=sbatch_args,
            script_args=script_args)
