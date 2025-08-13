import os
import argparse
import pathspec


def load_gitignore_patterns(repo_root):
    """
    Loads and parses the .gitignore file from the repository root.

    Args:
        repo_root (str): The path to the repository root.

    Returns:
        pathspec.PathSpec: A PathSpec object with the .gitignore patterns, or None if no .gitignore file exists.
    """
    gitignore_path = os.path.join(repo_root, '.gitignore')
    if os.path.isfile(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            return pathspec.PathSpec.from_lines('gitwildmatch', f)
    return None


def print_directory_contents(directory):
    """
    Prints the contents of all files in the specified directory and its subdirectories,
    ignoring files and directories matched by .gitignore patterns.

    Args:
        directory (str): The path to the directory to process.
    """
    # Ensure the directory exists
    if not os.path.isdir(directory) and not os.path.exists(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    # Get the repository root (assuming the script is run from the repo root)
    repo_root = os.getcwd()

    # Load .gitignore patterns
    gitignore_spec = load_gitignore_patterns(repo_root)

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        # Filter directories to exclude those matching .gitignore
        if gitignore_spec:
            dirs[:] = [d for d in dirs if
                       not gitignore_spec.match_file(os.path.relpath(os.path.join(root, d), repo_root))]

        for file_name in files:
            # Construct the full file path
            file_path = os.path.join(root, file_name)
            # Get the relative path from the repository root for .gitignore matching
            relative_path_from_root = os.path.relpath(file_path, repo_root)

            # Skip files that match .gitignore patterns
            if gitignore_spec and gitignore_spec.match_file(relative_path_from_root):
                continue

            # Get the relative path from the specified directory for printing
            relative_path_from_dir = os.path.relpath(file_path, directory)
            print(f"{relative_path_from_dir}:")
            try:
                # Read and print the file contents
                with open(file_path, 'r', encoding='utf-8') as file:
                    contents = file.read()
                    print(contents)
            except Exception as e:
                print(f"Error reading file: {e}")
            print()  # Add a blank line after each file's contents


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Print contents of all files in a directory and its subdirectories, respecting .gitignore.")
    parser.add_argument("directory", help="The directory to process")
    args = parser.parse_args()

    # Call the function with the provided directory
    print_directory_contents(args.directory)