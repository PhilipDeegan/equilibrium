FROM debian:latest

COPY sudo.sh /root
RUN cd /root && chmod +x sudo.sh && ./sudo.sh && adduser user

COPY user.sh /home/user
RUN cd /home/user && chmod +x user.sh && chown user user.sh

USER user
RUN cd /home/user && ./user.sh
