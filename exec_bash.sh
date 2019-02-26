#!/bin/bash

docker container prune -f
docker exec -ti file-connector /bin/sh
