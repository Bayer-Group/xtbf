doctests:
	pytest . --doctest-modules 

doc-html:
	export PDOC_ALLOW_EXEC=1
	pdoc --html --output-dir doc xtbf

doc-show:
	pdoc xtbf

install-deps:
	pip install -r requirements.txt
