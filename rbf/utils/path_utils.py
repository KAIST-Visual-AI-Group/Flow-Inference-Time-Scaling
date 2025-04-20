import os
import re
import glob

def gather_paths(pattern):
    """
    Generates a dictionary mapping key components to file paths based on a glob-like pattern with dynamic :n: placeholders.
    
    Args:
        pattern (str): A string pattern with placeholders like :n:, '*' for wildcards, 
                       and '?' for single-character matches.
    
    Returns:
        dict: A dictionary where the keys are tuples representing extracted key components 
              (based on :n:) and the values are the corresponding file paths.
    """

    # Dictionary to store the matching paths with their corresponding keys
    paths = {}

    # Create a regex pattern from the given pattern
    regex_pattern = ''
    glob_pattern = ''
    keys_indices = {}  # To store where :n: components are located
    key_count = 0  # Count the number of keys (:0:, :1:, etc.)
    
    i = 0
    while i < len(pattern):
        if pattern[i] == '*':
            # Handle *
            regex_pattern += r'[^/]+'  # Match any string except '/'
            glob_pattern += '*'  # Add * to the glob pattern
            key_count += 1
        elif pattern[i] == '?':
            # Handle ? (any single character)
            regex_pattern += r'([^/])'
            glob_pattern += '?'
            key_count += 1
        elif pattern[i] == ':':
            # Handle :n: (dynamic handling for any :n: pattern)
            j = i + 1
            while j < len(pattern) and pattern[j].isdigit():
                j += 1
            if j < len(pattern) and pattern[j] == ':':
                # Found a complete :n: pattern
                key_number = int(pattern[i+1:j])  # Extract the number in :n:
                regex_pattern += r'([^/]+)'  # Match anything except hyphen (-)
                glob_pattern += '*'  # Add * to the glob pattern
                keys_indices[key_count] = key_number
                key_count += 1
                i = j  # Skip ahead to after the closing ':'
            else:
                # If no proper closing ':', treat it as normal text
                regex_pattern += re.escape(pattern[i])
                glob_pattern += pattern
        elif pattern[i] == '$':
            # Handle $ (treat inner text as a valid regex expression)
            j = i + 1
            while j < len(pattern) and pattern[j] != '$':
                j += 1
            if j < len(pattern):
                # Found a complete $...$ pattern
                regex_pattern += pattern[i+1:j]  # Add the inner regex directly
                glob_pattern += '*'  # Add * to the glob pattern
                i = j  # Skip ahead to after the closing '$'
            else:
                # If no proper closing '$', treat it as normal text
                regex_pattern += re.escape(pattern[i])
                glob_pattern += pattern
        else:
            # Static text, needs exact match
            regex_pattern += re.escape(pattern[i])
            glob_pattern += pattern[i]
        i += 1
    
    # Compile the regex for matching the full paths
    regex_pattern = "^" + regex_pattern + "$"
    print(regex_pattern)  # Debugging purpose to show the regex pattern
    regex = re.compile(regex_pattern)
    
    # Use glob to collect candidate paths that match the glob-like part of the pattern
    candidate_paths = glob.glob(glob_pattern, recursive=True)

    # Iterate over candidate paths and match them against the regex
    counter = {}
    for path in candidate_paths:
        match = regex.match(path)
        if match:
            # Extract the key components
            key_components = tuple(match.groups()[keys_indices[i]] for i in sorted(keys_indices))
            cnt = counter.get(key_components, 0)
            counter[key_components] = cnt + 1
            paths[key_components + (cnt,)] = path
    
    return paths


def filter_paths(paths, *keys):
    """
    Filters a dictionary of paths by matching specified key components, with '*' as a wildcard.
    
    Args:
        paths (dict): A dictionary where keys are tuples of path components, and values are file paths.
        *keys (str): A variable number of key components to filter by. '*' can be used as a wildcard.
    
    Returns:
        dict: A filtered dictionary containing only the paths where the key components match the specified values.
    """
    # Dictionary to store filtered paths
    filtered_paths = {}

    # Check each path's key
    for path_key, path_value in paths.items():
        # Assume the key matches unless proven otherwise
        match = True
        for key_element, path_element in zip(keys, path_key):
            if key_element != '*' and key_element != path_element:
                match = False
                break
        if match:
            filtered_paths[path_key] = path_value

    return filtered_paths

def collect_keys(paths, idx):
    """
    Collects unique values at a specific index in the keys of the paths dictionary.
    
    Args:
        paths (dict): A dictionary where keys are tuples of path components, and values are file paths.
        idx (int): The index of the key component to collect across all paths.
    
    Returns:
        list: A list of unique values found at the specified index in the path keys.
    """
    keys = list(paths.keys())
    key_idx = list(set([key[idx] for key in keys]))
    return key_idx