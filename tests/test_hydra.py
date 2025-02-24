# stdlib
import os
from pathlib import Path

# third party
import pytest
import yaml
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig


def get_project_root() -> Path:
    """Returns project root folder."""
    # Get the directory containing the test file
    test_dir = Path(__file__).parent
    # Go up one level to reach project root
    return test_dir.parent


# Function to find all YAML files in a directory
def find_yaml_files(directory: Path):
    for path in directory.rglob("*"):
        print(f"Found: {path.relative_to(directory)}")

    yaml_files = []

    for path in directory.glob("*.yaml"):
        rel_path = os.path.relpath(str(path), directory)
        yaml_files.append(rel_path)
    return yaml_files


@pytest.fixture(params=find_yaml_files(get_project_root() / "conf"))
def hydra_config(request):
    config_dir = get_project_root() / "conf"

    # Initialize hydra with absolute path
    with initialize(config_path="../conf"):

        config_file = request.param

        # Use Path for reliable path joining
        config_path = config_dir / config_file

        # Check that the YAML file corresponds to a dictionary
        with config_path.open("r") as f:
            config_dict = yaml.safe_load(f)
            if not isinstance(config_dict, dict):
                pytest.skip(
                    f"Config file {config_file} does not correspond to a dictionary"
                )

        # Try to compose and instantiate the configuration
        config = compose(config_file)
        instantiate(config)

    return config


# Test function to ensure successful instantiation of Hydra configurations
def test_hydra_config_instantiation(hydra_config):
    assert isinstance(
        hydra_config, DictConfig
    ), f"Hydra config is not an DictConfig object but a {type(hydra_config)}"


def test_instanciate_pipeline(hydra_config):
    # Instantiate and train evaluators from config
    if "pipelines" not in hydra_config:
        pytest.skip("No pipelines found in configuration")

    for method, pipeline_cfg in hydra_config.pipelines.items():
        pipeline = instantiate(pipeline_cfg)

        pipeline.conformal_evaluator = pipeline.conformal_evaluator(
            conf_predictor_cfg=hydra_config.conf_predictor_cfg, logger=None
        )

        assert isinstance(pipeline.conformal_evaluator.conf_predictor_cfg, DictConfig)
