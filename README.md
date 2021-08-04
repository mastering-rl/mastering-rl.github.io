This project contains a Jupyter Book project for an introduction to reinforcement learning interactive book.

To build, first install https://jupyterbook.org/:

  pip install -U jupyter-book

You can then build with:

  bash build.bash

OR
  
  jupyter-book build .

from the current directory.

To publish changes to the website http://gibberblot.github.io/rl-notes/intro.html, you will need to install ghp-import using:

  pip install ghp-import

Before publishing, commit and push all changes to the master branch of the book. Then, build and publish with:

  bash publish.bash

OR just publish with:

  ghp-import -n -p -f _build/html

The webpage should update within 1-2 minutes
