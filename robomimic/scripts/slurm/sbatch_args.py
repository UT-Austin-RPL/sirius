# from slurm.util.arguments.base_args import *
from robomimic.scripts.slurm.base_args import *

PARTITIONS = (
    "napoli",
    "tibet",
    "svl",

    "titans",
    "dgx",
)


def add_sbatch_args():
    """
    Adds sbatch arguments needed for automatically generating and executing python files
    """
    # Define namespace for the robosuite args
    prefix = 'sbatch'
    actions = {
        "const": prefix,
        "action": GroupedAction
    }
    # Required args
    parser.add_argument(
        '--script',
        type=str,
        required=True,
        help='path to the Python script to execute',
        **actions
    )
    parser.add_argument(
        '--generated_dir',
        type=str,
        required=True,
        help='Sets the location where generated sbatch scripts will be stored',
        **actions
    )
    parser.add_argument(
        '--python_interpreter',
        type=str,
        required=True,
        help='Python interepreter to use for the executed python script',
        **actions
    )

    # Additional args
    parser.add_argument(
        '--partition',
        type=str,
        default='titans',
        choices=PARTITIONS,
        help='partition to run on for this process',
        **actions
    )
    parser.add_argument(
        '--exclude',
        type=str,
        default='',
        help='any specific machines to avoid, comma separated',
        **actions
    )
    parser.add_argument(
        '--gpu_type',
        type=str,
        default="any",
        help='Specific GPU to use. Any results in any GPU being used for this run',
        **actions
    )
    parser.add_argument(
        '--num_gpu',
        type=int,
        default=0,
        help='Sets the number of gpus to use for this sbatch script',
        **actions
    )
    parser.add_argument(
        '--num_cpu',
        type=int,
        default=4,
        help='Sets the number of cpus to use for this sbatch script',
        **actions
    )
    parser.add_argument(
        '--mem_gb',
        type=int,
        default=0,
        help='If nonzero, sets the amount of memory to be this many GB',
        **actions
    )
    parser.add_argument(
        '--max_hours',
        type=int,
        default=20,
        help='Sets the maximum number of hours this script will be run for',
        **actions
    )
    parser.add_argument(
        '--extra_pythonpath',
        type=str,
        default="",
        help='Extra paths to set to the pythonpath variable',
        **actions
    )
    parser.add_argument(
        '--overwrite',
        type=str,
        default="False",
        choices=BOOL_CHOICES,
        help='Whether to overwrite or not',
        **actions
    )
    parser.add_argument(
        '--extra_commands',
        nargs="+",
        type=str,
        default=None,
        help='Extra commands to run after main python command',
        **actions
    )
    parser.add_argument(
        '--copy_file',
        nargs="+",
        type=str,
        default=None,
        help='Copies a file from source to location. Expected format is [source_file, targeT_dir]. New file will'
             'share the same file name as the original source file. Useful in cases e.g.: copying datasets to local ssd',
        **actions
    )
    parser.add_argument(
        '--executable_log_dir',
        type=str,
        default='/cvgl2/u/jdwong/test_output',
        help='Location to dump sbatch log out / err text to',
        **actions
    )
    parser.add_argument(
        '--shell_source_script',
        type=str,
        default=None,
        help='If specified, bash script to source at beginning of sbatch execution',
        **actions
    )
    parser.add_argument(
        '--notification_email',
        type=str,
        default='jdwong@stanford.edu',
        help='Email address to send slurm notifications to (i.e.: when the script finishes running)',
        **actions
    )
    parser.add_argument(
        '--mujoco_dir',
        type=str,
        default='/cvgl2/u/jdwong/.mujoco/mujoco200/bin',
        help='Absolute path to mujoco 200 installation bin directory',
        **actions
    )
