#!/bin/bash

nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9 2> /dev/null
rm -f /dev/shm/fasterdp-shm-*
rm -f /dev/shm/sem.fasterdp-sem-*
