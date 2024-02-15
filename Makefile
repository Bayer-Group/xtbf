doctests:
	pytest . --doctest-modules 

doc-html:
	export PDOC_ALLOW_EXEC=1
	pdoc --html --output-dir doc xtbf

doc-show:
	pdoc xtbf

install-deps:
	pip install -r requirements.txt

download_datasets:
	mkdir data | echo "data dir exists"
	cd data && wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv
	cd data && wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv
	cd data && wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv
	cd data && wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv
	cd data && wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz && gzip -d toxcast_data.csv.gz
