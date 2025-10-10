#!/usr/bin/env python3
import os
import sys


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)

    base_dir = sys.argv[1]
    if not os.path.isdir(base_dir):
        print(f"{base_dir} is not a directory")
        sys.exit(1)

    subdirs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    subdirs.sort()

    readme_path = os.path.join(base_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Data Reports\n\n")
        for d in subdirs:
            f.write(f"- [{d}]({d}/)\n")


if __name__ == "__main__":
    main()
