import argparse
import sys

def run_lateral():
    from lateral.inference import run
    run()

def run_ventral():
    from ventral.inference import run
    run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--view",
        required=True,
        choices=["lateral", "ventral"],
        help="Select inference view"
    )
    args = parser.parse_args()

    if args.view == "lateral":
        run_lateral()
    elif args.view == "ventral":
        run_ventral()
    else:
        sys.exit("Invalid view selected")
