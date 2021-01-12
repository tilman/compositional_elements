# compoelem
Library for generating and comparing compositional elements from art historic images.

## Build and Setup:
```bash
git clone https://github.com/tilman/compoelem
cd compoelem
git submodule update --init --recursive
git submodule update --recursive
```
### Build and Setup Openpose
Installation routine could change with an update of the git submodule. Tested with git commit b3e8abf from 20 Dec 2019. For an up to date installation procedure visit: https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation
Depends on swig therfore run `apt install swig` for ubuntu or `brew install swig` for Mac OSX.
Also make sure to download the model and place it under `compoelem/compoelem/detect/openpose/pose_model.pth`
```bash
pip install -r requirements.txt
cd compoelem/detect/openpose
cd lib/pafprocess; sh make.sh
```
## Installation of the library:
```bash
python setup.py clean --all
python setup.py bdist_wheel
pip install dist/compoelem-0.1.0-py3-none-any.whl
```
or
```bash
python setup.py clean --all
python setup.py install
```

## Usage:
See [docs](https://tilman.github.io/compoelem/compoelem/) and [tests](tests/test_e2e.py) for example usage.

## Run Test Suite:
`python setup.py pytest`

## Build the documentation:
`pip install pdoc3`
`rm -rf docs && pdoc3 --html --output-dir docs --config show_source_code=False compoelem`
