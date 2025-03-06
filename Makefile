# cp ../Makefile Makefile
#import sys

#!"{sys.executable}" --version
#!which "{sys.executable}"
#!jupyter labextension list

-include .env
export

jupyter: 
	@uv run jupyter lab


install:
	@uv add jupyterlab-vim jupyterlab-code-formatter ipywidgets black isort
	@uv add numpy pandas matplotlib seaborn scikit-learn


render: create_docs_dir
	uv run quarto render

# Similar to convert, but only convert the diff files.
# You need to 'git add .' the files before running this command.
render_diff: create_docs_dir
	$(foreach file, $(shell git diff HEAD --name-only | grep .ipynb), uv run quarto render $(file) --to gfm --output-dir docs/)


create_docs_dir:
	mkdir -p docs


lint:
	@uv run black *.py

# uv run quarto create project
