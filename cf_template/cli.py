import argparse

def cli():
    """"""
    description = "Cognite Functions Template CLI"

    epilog = """This CLI provides an entry point to the
Cognite Functions Template.
"""

    parser = argparse.ArgumentParser(
        description=description,
        add_help=True,
        epilog=epilog
    )

    parser.add_argument("-n", "--name", type=str, help="Name of the new cognite function project.")
    return parser
