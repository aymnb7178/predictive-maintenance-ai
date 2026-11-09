.PHONY: install test demo clean

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

demo:
	python scripts/demo.py

clean:
	rm -rf build models .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
