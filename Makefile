install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python3 -m pytest

format:
	black *.py


lint:

	-python3 -m pylint --disable=R,C *.py

all: install lint test
