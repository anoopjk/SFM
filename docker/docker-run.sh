#!/bin/bash

# Enable X-forwarding  
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    /usr/bin/xhost +
fi

# Run the docker container
docker run -p 6006:6006 -p 8888:8888 --rm -it --gpus all --entrypoint /bin/bash  \
    --name sfm \
    -v "$(pwd)"/../:/SFM \
    -v /home/$USER:/home/$USER \
    -v /media:/media \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    docker_sfm:latest