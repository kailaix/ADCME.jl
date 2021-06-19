FROM tensorflow/tensorflow:1.15.5-py3
MAINTAINER Kailai Xu kailaix@hotmail.com

ENV DOCKER_BUILD=1
ENV PYTHON=/usr/local/bin/python
RUN apt update && apt  -y upgrade && \
    apt install  -y wget vim ninja-build cmake unzip libopenblas-dev
RUN cd ~ && \ 
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz && \
    tar -xzf julia-1.6.1-linux-x86_64.tar.gz && \ 
    echo "export PATH=$PATH:~/julia-1.6.1/bin" >> ~/.bashrc
RUN /usr/local/bin/pip install scipy tensorflow-probability==0.8 matplotlib
RUN echo "export DOCKER_BUILD=1" >> ~/.bashrc
RUN echo "export PYTHON=/usr/local/bin/python" >> ~/.bashrc
RUN ~/julia-1.6.1/bin/julia -e "using Pkg; Pkg.add(\"PyCall\")"
RUN ~/julia-1.6.1/bin/julia -e "using Pkg; ENV[\"DOCKER_BUILD\"] = 1; Pkg.add(PackageSpec(name=\"ADCME\", rev=\"master\"))"
RUN ~/julia-1.6.1/bin/julia -e "using ADCME; ADCME.precompile()"
CMD ~/julia-1.6.1/bin/julia