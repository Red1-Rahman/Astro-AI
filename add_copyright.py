#!/usr/bin/env python3
"""
Copyright Header Tool for Astro-AI Platform
Adds copyright notices to all Python files in the project.
"""

import os
import glob
from pathlib import Path

COPYRIGHT_HEADER = '''"""
Astro-AI: Galaxy Evolution Analysis Platform
Copyright (c) 2025 Redwan Rahman and CAM-SUST

All rights reserved. Licensed under the Astro-AI Proprietary License.
See LICENSE file for full terms and conditions.

Contact: redwanrahman2002@outlook.com
"""

'''


def add_copyright_header(file_path):
    """Add copyright header to a Python file if not already present."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Skip if copyright already exists
        if "Copyright (c) 2025 Redwan Rahman and CAM-SUST" in content:
            print(f"Copyright already exists in {file_path}")
            return

        # Skip if file starts with shebang
        if content.startswith("#!"):
            lines = content.split("\n")
            # Find the first non-shebang, non-comment line
            insert_pos = 0
            for i, line in enumerate(lines):
                if not line.startswith("#") and line.strip():
                    insert_pos = i
                    break

            # Insert copyright after shebang and initial comments
            lines.insert(insert_pos, COPYRIGHT_HEADER.rstrip())
            new_content = "\n".join(lines)
        else:
            # Add copyright at the beginning
            new_content = COPYRIGHT_HEADER + content

        # Write the updated content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"Added copyright header to {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    """Add copyright headers to all Python files in the project."""
    project_root = Path(__file__).parent
    python_files = glob.glob(str(project_root / "**/*.py"), recursive=True)

    # Skip this script itself and __pycache__ files
    python_files = [
        f
        for f in python_files
        if not f.endswith("add_copyright.py") and "__pycache__" not in f
    ]

    print(f"Found {len(python_files)} Python files to process...")

    for file_path in python_files:
        add_copyright_header(file_path)

    print("\nCopyright header addition complete!")
    print("\nIMPORTANT: Review all files before committing to ensure")
    print("copyright headers are properly placed and don't break functionality.")


if __name__ == "__main__":
    main()
