# compositional_elements
Library for generating and comparing compositional elements from art historic images.

## Installation of the library:
```
python setup.py bdist_wheel
pip install dist/compositional_elements-0.1.0-py3-none-any.whl
```
or
```
python setup.py install
```

## Usage:
See [docs](https://tilman.github.io/compositional_elements/compositional_elements/) and [tests](tests/generate/test_pose_direction.py) for example usage.

## Run Test Suite:
`python setup.py pytest`

## Build the documentation:
`rm -rf docs && pdoc --html --output-dir docs --config show_source_code=False compositional_elements`
