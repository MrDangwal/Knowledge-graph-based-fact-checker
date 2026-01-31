PYTHON=python3.10
VENV=.venv

setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate; pip install --upgrade pip; pip install -r requirements.txt

run:
	. $(VENV)/bin/activate; uvicorn app.main:app --reload

test:
	. $(VENV)/bin/activate; pytest -q
