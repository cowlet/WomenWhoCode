# Installation notes for running these examples

These notes describe how to install the software needed for the workshop "Image Classification using Deep Neural Networks" run by [Women Who Code Toronto][wwct].

[wwct]:     https://www.meetup.com/Women-Who-Code-Toronto/

To run the examples in the workshop, we need to install the following:

- Python 3
- Numpy
- Scipy
- TensorFlow
- Keras

## If you already have Python 3 installed

If you have previously installed Python 3 using Homebrew or another package manager, the libraries should be installable using the following commands:
- `pip3 install numpy`
- `pip3 install scipy`
- `pip3 install tensorflow`
- `pip3 install keras`

Some of these might take some time to download and install.

## If you don't know if you have Python 3 or not

Check by opening a Terminal and running `python3`. If you get a prompt (`>>>`), congrats! You already have it. Try running all the commands above to make sure you have all the libraries.

## If you don't have Python 3

If you don't have Python 3 installed, you can get it through either the [Anaconda package manager][anaconda] (more graphical/user friendly) or Homebrew (run `brew install python3` in a Terminal). 

[anaconda]:     https://www.continuum.io/downloads

Next, install the libraries. If you chose Homebrew, run the commands above. If using Anaconda, run:

- `conda install numpy`
- `conda install scipy`
- `pip3 install tensorflow`
- `pip3 install keras`

## Notes and Comments

- TensorFlow can be installed for "CPU only", or "Using GPU" to speed up computation. The command here goes for "CPU only", which is fine (I use it on a MacBook Pro from 2014 without problems).
- If you have Python 2 and want to keep using it, no problem! Change all the install commands to `pip` instead of `pip3`, and you may have to adapt the syntax of the examples a little. 

## Testing that everything works

To test that everything is installed ok, open up a Python prompt (either in Anaconda, or in a Terminal run `python3`), and paste in `import keras`. If everything is installed correctly, you should see something like this:

    >>> import keras
    Using TensorFlow backend.

