FROM anoop/pytorch:latest

# create symlink of python3 as python so
# that you don't have to use python3 command everytime
RUN ln -sv /usr/bin/python3 /usr/bin/python
RUN pip install jupyter


RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'

RUN pip install open3d \
				pandas  \
				addict  \
				plyfile  \
				tqdm 


# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"



