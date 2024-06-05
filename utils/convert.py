import jsonlines
from tqdm import tqdm
import argparse
import os

source = ''
target = ''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=str,
        default="",
        help="input file",
    )
    parser.add_argument(
        "output",
        type=str,
        default="",
        help="output file",
    )
    parser.add_argument(
        "--dev",
        default=False,
        action='store_true',
        help="whether dev",
    )

    args = parser.parse_args()
    input_file, output_file = args.input, args.output
    mode = 'dev' if args.dev else 'train'

    with jsonlines.open(input_file) as reader:
        with jsonlines.open(os.path.join(output_file, mode + '.source'), 'w') as source:
            with jsonlines.open(os.path.join(output_file, mode + '.target'), 'w') as target:
                for item in tqdm(reader):
                    source.write(item['source'])
                    target.write(item['target'])