This project contains a Jupyter Book project for an introduction to reinforcement learning interactive book.

# 1. Installing dependencies
- Install required libraries to make sure the code can run properly in the notebook
```
conda env create -f environment.yml
conda activate rlnotes
```
- If you update environment.yml with new dependencies, you can run this to update the conda environment
```
conda activate rlnotes
conda env update --file environment.yml --prune
```
- Include all python codes to execute code cells
```
export PYTHONPATH="$PWD/mastering_rl"
```

# 2. Building the project
To build, first install https://jupyterbook.org/: (No need to do this if you already installed the conda env `rlnotes` successfully)
```
pip install -U jupyter-book
```

- You can then build with:
```
bash build.bash
```
OR
```
jupyter-book build .
```
from the current directory.

- OR You can do a full re-build by cleaning all cache files before building the project
```
bash clean-build.bash
```

# 3. Publishing the project
To publish changes to the website http://gibberblot.github.io/rl-notes/intro.html, you will need to install ghp-import using: (No need to do this if you already installed the conda env `rlnotes` successfully)
```
pip install ghp-import
```

Before publishing, commit and push all changes to the master branch of the book. Then, build and publish with:
```
bash publish.bash
```

OR just publish with:
```
ghp-import -n -p -f _build/html
```

The webpage should update within 1-2 minutes
