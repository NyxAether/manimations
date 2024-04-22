black manimations tests
autoflake -r --in-place manimations tests
isort manimations tests
mypy manimations tests
flake8 manimations tests
pylint manimations tests