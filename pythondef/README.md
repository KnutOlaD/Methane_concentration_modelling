# Setup
Setting up a singularity container you can use to run OpenDrift:

Create a singularity image from a linux terminal on a machine where you have sudo privileges, and type:
```
sudo singularity build python.sif Singularity.def
```
You will need `Singularity.def` and `requirements.txt` in the directory from where you build this image. 

# Usage
Activate the image by calling;
```
singularity shell python.sif
```

Bind paths and mounts to the image, like this on sigma2`clusters
```
singularity shell --bind /cluster python.sif
```

Once having activated the singularity shell, you're effectively running a Ubuntu machine with python and all python modules necessary to use fvtools interatively on the cluster. I like to work in ipyton, and this simply just need to call it to get started;
```singularity
ipython
```

Or, alternatively start a jupyter lab notebook
```singularity
jupyter lab --no-browser --port 8890
```

create a link to that port from, say, betzy to your local machine
```bash
ssh -N -f -L localhost:8890:localhost:8890 hes001@betzy.sigma2.no
```
and you should be set to go.
