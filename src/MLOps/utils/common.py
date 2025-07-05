import os
import sys
import yaml
import json
import joblib
from pathlib import Path
from typing import Any, Union, Dict, List
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read YAML file and return its contents as a ConfigBox object.
    
    Args:
        path_to_yaml (Path): Path to the YAML file
        
    Returns:
        ConfigBox: Contents of the YAML file as a ConfigBox object
        
    Raises:
        ValueError: If the YAML file is empty
        FileNotFoundError: If the YAML file does not exist
        yaml.YAMLError: If there's an error parsing the YAML file
        
    Examples:
        >>> config = read_yaml(Path("config/config.yaml"))
        >>> print(config.data_ingestion.source_url)
    """
    try:
        with open(path_to_yaml, 'r', encoding='utf-8') as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError(f"YAML file '{path_to_yaml}' is empty")
            return ConfigBox(content)
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found: {path_to_yaml}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file '{path_to_yaml}': {e}")
    except BoxValueError as e:
        raise ValueError(f"Error creating ConfigBox from YAML content: {e}")


@ensure_annotations
def load_json(path_to_json: Path) -> Dict[str, Any]:
    """
    Load JSON file and return its contents as a dictionary.
    
    Args:
        path_to_json (Path): Path to the JSON file
        
    Returns:
        Dict[str, Any]: Contents of the JSON file as a dictionary
        
    Raises:
        FileNotFoundError: If the JSON file does not exist
        json.JSONDecodeError: If there's an error parsing the JSON file
        
    Examples:
        >>> data = load_json(Path("data/sample.json"))
        >>> print(data["key"])
    """
    try:
        with open(path_to_json, 'r', encoding='utf-8') as json_file:
            content = json.load(json_file)
            return content
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path_to_json}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error parsing JSON file '{path_to_json}': {e}", 
                                 e.doc, e.pos)


# @ensure_annotations
def create_directory(path_to_directory: Union[Path, str], verbose: bool = True) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path_to_directory (Union[Path, str]): Path to the directory to create
        verbose (bool, optional): Whether to print creation message. Defaults to True.
        
    Examples:
        >>> create_directory("artifacts/data_ingestion")
        >>> create_directory(Path("models/trained"), verbose=False)
    """
    path_to_directory = Path(path_to_directory)
    
    if not path_to_directory.exists():
        path_to_directory.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Created directory: {path_to_directory}")
    elif verbose:
        print(f"Directory already exists: {path_to_directory}")

def save_json(path_to_json, data, indent: int = 4) -> None:
    """
    Save dictionary as JSON file.
    
    Args:
        path_to_json (Path): Path where the JSON file will be saved
        data (Dict[str, Any]): Dictionary data to save
        indent (int, optional): Number of spaces for indentation. Defaults to 4.
        
    Raises:
        TypeError: If data is not JSON serializable
        PermissionError: If unable to write to the specified path
        
    Examples:
        >>> data = {"accuracy": 0.95, "loss": 0.05}
        >>> save_json(Path("metrics/results.json"), data)
    """
    path_to_json = Path(path_to_json)
    try:
        # Create directory if it doesn't exist
        create_directory(path_to_json.parent, verbose=False)
        
        with open(path_to_json, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=indent, ensure_ascii=False)
        print(f"JSON data saved to: {path_to_json}")
    except TypeError as e:
        raise TypeError(f"Data is not JSON serializable: {e}")
    except PermissionError:
        raise PermissionError(f"Permission denied: Cannot write to {path_to_json}")


@ensure_annotations
def save_bin(data: Any, path_to_bin: Path) -> None:
    """
    Save data as binary file using joblib.
    
    Args:
        data (Any): Data to save (models, arrays, objects, etc.)
        path_to_bin (Path): Path where the binary file will be saved
        
    Raises:
        PermissionError: If unable to write to the specified path
        
    Examples:
        >>> model = train_model()
        >>> save_bin(model, Path("models/trained_model.joblib"))
        >>> 
        >>> array = np.array([1, 2, 3, 4, 5])
        >>> save_bin(array, Path("data/processed_array.pkl"))
    """
    try:
        # Create directory if it doesn't exist
        create_directory(path_to_bin.parent, verbose=False)
        
        joblib.dump(value=data, filename=path_to_bin)
        print(f"Binary data saved to: {path_to_bin}")
    except PermissionError:
        raise PermissionError(f"Permission denied: Cannot write to {path_to_bin}")
    except Exception as e:
        raise Exception(f"Error saving binary data to {path_to_bin}: {e}")


@ensure_annotations
def load_bin(path_to_bin: Path) -> Any:
    """
    Load data from binary file using joblib.
    
    Args:
        path_to_bin (Path): Path to the binary file
        
    Returns:
        Any: Loaded data object
        
    Raises:
        FileNotFoundError: If the binary file does not exist
        
    Examples:
        >>> model = load_bin(Path("models/trained_model.joblib"))
        >>> predictions = model.predict(X_test)
        >>> 
        >>> array = load_bin(Path("data/processed_array.pkl"))
        >>> print(array.shape)
    """
    try:
        data = joblib.load(path_to_bin)
        print(f"Binary data loaded from: {path_to_bin}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Binary file not found: {path_to_bin}")
    except Exception as e:
        raise Exception(f"Error loading binary data from {path_to_bin}: {e}")


@ensure_annotations
def get_size(path: Union[str|Path]) -> str:
    """
    Get size of file or directory in human-readable format.
    
    Args:
        path (Path): Path to file or directory
        
    Returns:
        str: Size in human-readable format (B, KB, MB, GB, TB)
        
    Raises:
        FileNotFoundError: If the path does not exist
        
    Examples:
        >>> size = get_size(Path("data/dataset.csv"))
        >>> print(f"Dataset size: {size}")
        >>> 
        >>> dir_size = get_size(Path("artifacts/"))
        >>> print(f"Artifacts directory size: {dir_size}")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    if path.is_file():
        size_bytes = path.stat().st_size
    else:
        # Calculate directory size
        size_bytes = sum(
            file.stat().st_size 
            for file in path.rglob('*') 
            if file.is_file()
        )
    
    # Convert bytes to human-readable format
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} PB"


# Additional utility functions for common MLOps tasks

@ensure_annotations
def get_file_extension(file_path: Path) -> str:
    """
    Get file extension from file path.
    
    Args:
        file_path (Path): Path to the file
        
    Returns:
        str: File extension (without the dot)
        
    Examples:
        >>> ext = get_file_extension(Path("data/train.csv"))
        >>> print(ext)  # Output: csv
    """
    return file_path.suffix.lstrip('.')


# @ensure_annotations
def create_directories(list_of_directories: List[Union[Path, str]], verbose: bool = True) -> None:
    """
    Create multiple directories at once.
    
    Args:
        list_of_directories (List[Union[Path, str]]): List of directory paths to create
        verbose (bool, optional): Whether to print creation messages. Defaults to True.
        
    Examples:
        >>> dirs = ["artifacts/data_ingestion", "artifacts/model_training", "logs"]
        >>> create_directories(dirs)
    """
    for directory in list_of_directories:
        create_directory(directory, verbose=verbose)