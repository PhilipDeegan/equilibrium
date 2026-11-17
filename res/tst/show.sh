
# docker run --rm --net=host -u $(id -u ${USER}):$(id -g ${USER})  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -h $HOSTNAME guitest:1.0
