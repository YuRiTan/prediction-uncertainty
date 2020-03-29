# Code used in blog(s)
- [A frequentist approach to prediction uncertainty](https://www.yuritan.nl/posts/prediction_uncertainty/)
 - [Prediction uncertainty - part 2](https://www.yuritan.nl/posts/prediction_uncertainty_part_2/)

## Getting started
There are basically two options when running this code. Either you run it locally using `Python 3(.7)` and `virtualenv`, or you spin up a docker container which mounts the project in the container. This way you can develop locally while the code is executed inside the docker container.

### Prerequisites
Run on local macine:
- Python 3 (3.7)
- virtualenv
Run containerized
- docker
- docker-compose

### Usage
To build the docker image and run jupter lab, you can use the following command:

``` bash
make jupyter
```

If you want to run it differently, please checkout the `Makefile` for other options.
