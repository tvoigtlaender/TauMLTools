from XRootD import client
import fnmatch
from urllib.parse import urlparse
import os
from glob import glob


def check_if_local(pattern):
    """
    Check if the given pattern is a local path.

    Args:
        pattern (str): The path pattern to check.

    Returns:
        bool: True if the pattern is local, False otherwise.
    """
    result = urlparse(pattern)
    # Check if the URL has a scheme and netloc, indicating a remote path
    if result.scheme and result.netloc:
        return False
    # Check if the URL has no scheme and no netloc, indicating a local path
    elif not result.scheme and not result.netloc:
        return True
    else:
        # Raise an error if the pattern is neither clearly remote nor local
        raise ValueError(f"Pattern {pattern} is neither fully remote nor local.") 


def recursive_glob(xrd_client, base_path, pattern_parts):
    """
    Recursively glob remote files matching the given pattern parts.

    Args:
        xrd_client (client.FileSystem): The XRootD client for remote operations.
        base_path (str): The base path to start the globbing.
        pattern_parts (list): List of pattern parts to match.

    Returns:
        list: List of matched file paths.
    """
    if not pattern_parts:
        return []

    current_pattern = pattern_parts[0]
    remaining_patterns = pattern_parts[1:]

    try:
        # Check if the current pattern contains a glob character
        if '*' in current_pattern or '?' in current_pattern or '[' in current_pattern:
            # List the directory contents at the base path
            status, dir_list = xrd_client.dirlist(base_path)
            if not status.ok:
                # Print an error message if the directory listing fails
                print(f"Error listing directory {base_path}: {status.message}")
                return []

            # Filter the directory entries using the current pattern
            matching_entries = fnmatch.filter([entry.name for entry in dir_list.dirlist], current_pattern)
        else:
            # No globbing, so just use the current pattern as the directory name
            matching_entries = [current_pattern]

        # If there are no more remaining patterns, return the matched paths
        if not remaining_patterns:
            return [os.path.join(base_path, match) for match in matching_entries]

        matched_paths = []
        for match in matching_entries:
            full_match_path = os.path.join(base_path, match)
            # Recursively call the function for the remaining patterns
            matched_paths.extend(recursive_glob(xrd_client, full_match_path, remaining_patterns))

        return matched_paths
    except Exception as e:
        # Print an error message if any exception occurs
        print(f"Error processing {base_path} with pattern {current_pattern}: {e}")
        return []


def remote_glob(pattern):
    """
    Glob files for a given pattern, handling both local and remote paths.

    Args:
        pattern (str): The file path pattern to match.

    Returns:
        list: List of matched file paths.
    """
    if check_if_local(pattern):
        # Use the glob module for local path patterns
        return glob(pattern)
    else:
        # Parse the remote directory URL
        parsed_url = urlparse(pattern)

        # Extract the client URL and the file path
        client_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        file_path = parsed_url.path

        # Split the file path into parts for recursive globbing
        path_parts = file_path.strip('/').split('/')

        # Initialize the XRootD client
        xrd_client = client.FileSystem(client_url)

        # Perform recursive globbing
        matched_files = recursive_glob(xrd_client, '/', path_parts)

        # Return globbed remote files with client URL prefix
        return [f"{client_url}/{file_}" for file_ in matched_files]