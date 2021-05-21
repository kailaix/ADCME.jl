# Install ADCME Docker Image

For users who do not want to deal with installing and debugging dependencies, we also provide a [Docker](https://docs.docker.com/get-docker/) image for ADCME. Docker creates a virtual environment that isolates ADCME and its dependencies from the rest of the system. To install ADCME through docker, the user system must 

1. have Docker installed;
2. have sufficient space (at least 5G).

After docker has been installed, ADCME can be installed and launched via the following line (users do not have to install Julia separately because the Docker container is shipped with a compatible Julia binary)

```julia
docker run -ti kailaix/adcme 
```

This will fire a Julia prompt, which already includes precompiled ADCME. 

For users who want to open a terminal, run 

```julia
docker run -ti kailaix/adcme bash
```

This will launch a terminal where users can type `julia` to open a Julia prompt. 

## Tips

* To detach from the docker environment without suspending the process, press `Ctrl-p Ctrl-q`. To re-attach the process, first find the corresponding container ID

```bash
docker ps -a
```

Then 
```bash
docker container attach <container_id>
```

Or start a new bash from the same container

```
docker exec -ti <container_id> bash
```

* To share a folder with the host file system (e.g., share persistent data), use `-v` to attach a volume:

```bash
docker run -ti -v "/path/to/host/folder:/path/to/docker/folder" kailaix/adcme bash
```

* Create a new docker image containing all changes:

```bash
docker commit <container_id> <tag>
```

* Build and upload (change `kailaix/adcme` to your own repository)

```bash
docker build -t kailaix/adcme . 
docker run -ti kailaix/adcme
docker push kailaix/adcme
```

The current directory should have a Dockerfile:
```
FROM adcme:<tag>
CMD ["/julia-1.6.1/bin/julia"]
```

* Clean up docker containers
```bash
docker system prune 
```