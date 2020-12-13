# compositional_elements
Library for generating and comparing compositional elements from art historic images.

## Installation:
```
python setup.py bdist_wheel
pip install dist/compositional_elements-0.1.0-py3-none-any.whl
```
or
```
python setup.py install
```

## Usage:
See doc and tests for example usage.

## Run Test Suite:
`python setup.py pytest`

## Build the documentation:
`rm -rf doc && pdoc --html --output-dir doc --config show_source_code=False compositional_elements`