"""
File holding all command line arguments to use
"""

from argparse import ArgumentParser, Namespace, Action, ArgumentError, SUPPRESS, _UNRECOGNIZED_ARGS_ATTR
import sys as _sys

BOOL_CHOICES = ['True', 'False', 'true', 'false']
BOOL_MAPPING = {
    "false": False,
    "true": True
}
BOOL_STR = BOOL_MAPPING.keys()


def maybe_array_to_element(inp):
    """
    Maybe converts an array to a single (numerical) element. If len(inp) == 1, returns the input's first
    element. Otherwise, returns the input
    """
    return inp[0] if type(inp) is list and len(inp) == 1 else inp


# Define custom parsing class for nested default parses
class NestedParser(ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        if args is None:
            # args default to the system args
            args = _sys.argv[1:]
        else:
            # make sure that args are mutable
            args = list(args)

        # default Namespace built from parser defaults
        if namespace is None:
            namespace = Namespace()

        # add any action defaults that aren't present
        for action in self._actions:
            if action.dest is not SUPPRESS:
                if not hasattr(namespace, action.dest):
                    if action.default is not SUPPRESS:
                        # Send attribute to groupspace, not namespace!
                        groupspace = getattr(namespace, action.const, None) if action.const else namespace
                        if groupspace is None:
                            # Create new attribute in main namespace and reference this with groupspace
                            setattr(namespace, action.const, Namespace())
                            groupspace = getattr(namespace, action.const)
                        default = BOOL_MAPPING[action.default.lower()] \
                            if type(action.default) is str and action.default.lower() in BOOL_STR \
                            else action.default
                        setattr(groupspace, action.dest, default)

        # add any parser defaults that aren't present
        for dest in self._defaults:
            if not hasattr(namespace, dest):
                #groupspace = getattr(namespace, dest.const, Namespace()) if dest.const else namespace
                setattr(namespace, dest, self._defaults[dest])

        # parse the arguments and exit if there are any errors
        try:
            namespace, args = self._parse_known_args(args, namespace)
            if hasattr(namespace, _UNRECOGNIZED_ARGS_ATTR):
                args.extend(getattr(namespace, _UNRECOGNIZED_ARGS_ATTR))
                delattr(namespace, _UNRECOGNIZED_ARGS_ATTR)
            return namespace, args
        except ArgumentError:
            err = _sys.exc_info()[1]
            self.error(str(err))


# Define class for creating custom nested namespaces
class GroupedAction(Action):

    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None,
                 maybe_array=False,
                 ):
        # Add custom attributes
        self.maybe_array = maybe_array

        # Run super init
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        groupspace = getattr(namespace, self.const, Namespace())
        if type(values) is str and values.lower() in BOOL_STR:
            values = BOOL_MAPPING[values.lower()]
        # Possibly convert array if requested
        if self.maybe_array:
            values = maybe_array_to_element(values)
        setattr(groupspace, self.dest, values)
        setattr(namespace, self.const, groupspace)


# Define global parser
parser = NestedParser(description='Top level arguments')

# Add seed arg always
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')


# def parse_arguments():
#     """
#     Parses all arguments and splits them into their appropriate namespaces, returning separately the robosuite args,
#     rllib args, and agent args
#     """
#     args = parser.parse_args()
#     robosuite_args = getattr(args, "robosuite", None)
#     rllib_args = getattr(args, "rllib", None)
#     model_args = getattr(args, "model", None)
#     agent_args = getattr(args, "agent", None)
#
#     # Print all args
#     print()
#     for t, arg in zip(("robosuite", "rllib", "model", "agent"), (robosuite_args, rllib_args, model_args, agent_args)):
#         print('  {} Params: '.format(t))
#         if arg is not None:
#             for key, value in arg.__dict__.items():
#                 if key.startswith('__') or key.startswith('_'):
#                     continue
#                 print('    {}: {}'.format(key, value))
#         print()
#
#     # Return args
#     return robosuite_args, rllib_args, model_args, agent_args


if __name__ == '__main__':
    # Add arguments
    # add_robosuite_arguments()
    # add_rllib_arguments()
    # add_ppo_arguments()
    #
    # # Test parsing functionality
    # a, b, c = parse_arguments()
    # print(a)
    # print(b)
    # print(c)
    pass
