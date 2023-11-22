# YOLO-NAS Object Detection
## Prerequisites
### Pyenv
We recommend on managing your python versions via `pyenv`. <br>
To install pyenv on your specific OS, please refer to this link: **[pyenv-installation](https://github.com/pyenv/pyenv#installation)** <br>
After installing and setting up the shell environment you can install any python version, for this project it is recommended to install python `3.9.16`
```
pyenv install 3.9.16
```
After installation run the following from a shell within the project root `yolo-nas-coco`
```
pyenv local 3.9.16
```
this will set your local python to the specified version
### Poetry 
We recommend using poetry as your python dependency manager, and we supplied an environment defined by the `poetry.lock` file. <br>
To install poetry on your specific OS please refer to this link: **[poetry-installation](https://python-poetry.org/docs/#installing-with-the-official-installer)** <br>
when poetry is installed, run the following commands from within the project root folder `yolo-nas-coco` to set the environment python version and create the environment:
```
poetry env use 3.9.16
poetry install
```

## Model Import
To import the model into `Tensorleap`, first place it under the `model` directory.
Then run the following commands from a shell within the project root directory `yolo-nas-coco`:
```
sudo chmod +x upload_models.sh
```
make sure your poetry env is activated by running:
```commandline
poetry shell
```
and then run the script
```
./upload_models.sh model/<your-model-name>
```
the `upload_models.sh` script will import your raw model as well as a version of it with permuted 
outputs as well as the corresponding mapping so that you are ready to start evaluating 


# Getting Started with Tensorleap Project

This quick start guide will walk you through the steps to get started with this example repository project.

## Prerequisites

Before you begin, ensure that you have the following prerequisites installed:

- **[Python](https://www.python.org/)** (version 3.7 or higher).
- **[Poetry](https://python-poetry.org/)**.
- **[Tensorleap](https://tensorleap.ai/)** platform access. To request a free trial click [here](https://meetings.hubspot.com/esmus/free-trial).
- **[Tensorleap CLI](https://github.com/tensorleap/leap-cli)**.


## Tensorleap **CLI Installation**

with `curl`:

```
curl -s https://raw.githubusercontent.com/tensorleap/leap-cli/master/install.sh | bash
```

## Tensorleap CLI Usage

### Tensorleap **Login**

To login to Tensorleap:

```
tensorleap auth login [api key] [api url].
```

- API Key is your Tensorleap token (see how to generate a CLI token in the section below).
- API URL is your Tensorleap environment URL: https://api.CLIENT_NAME.tensorleap.ai/api/v2

<br>

**How To Generate CLI Token from the UI**

1. Login to the platform in 'CLIENT_NAME.tensorleap.ai'
2. Scroll down to the bottom of the **Resources Management** page, then click `GENERATE CLI TOKEN` in the bottom-left corner.
3. Once a CLI token is generated, just copy the whole text and paste it into your shell.


## Tensorleap **Project Deployment**

To deploy your local changes:

```
leap project push
```

### **Tensorleap files**

Tensorleap files in the repository include `leap_binder.py` and `leap.yaml`. The files consist of the  required configurations to make the code integrate with the Tensorleap engine:

**leap.yaml**

leap.yaml file is configured to a dataset in your Tensorleap environment and is synced to the dataset saved in the environment.

For any additional file being used, we add its path under `include` parameter:

```
include:
    - leap_binder.py
    - cityscapes_od/configs.py
    - [...]
```

**leap_binder.py file**

`leap_binder.py` configures all binding functions used to bind to Tensorleap engine. These are the functions used to evaluate and train the model, visualize the variables, and enrich the analysis with external metadata variables

## Testing

To test the system we can run `leap_test.py` file using poetry:

```
poetry run test
```

This file will execute several tests on leap_binder.py script to assert that the implemented binding functions: preprocess, encoders,  metadata, etc.,  run smoothly.

*For further explanation please refer to the [docs](https://docs.tensorleap.ai/)*


