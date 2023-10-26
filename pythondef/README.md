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

Once having activated the singularity shell, you're effectively running a Ubuntu machine with python and all python modules necessary to use OpenDrift interatively on the cluster. 

Those who loke to work in ipyton just need to call it to get started;
```singularity
ipython
```

Others might want to use a jupyter lab notebook
```singularity
jupyter lab --no-browser --port 8890
```

create a link to that port from, say, betzy to your local machine
```bash
ssh -N -f -L localhost:8890:localhost:8890 hes001@betzy.sigma2.no
```
and you should be set to go.

# Copy
The singularity image is essentially just an ubuntu-desktop with python pre-installed. You can use images made by others, and copy them to any machine you may want to use it on.
