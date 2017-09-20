FROM ubuntu:16.04

# Install dependencies, create "elm" user
RUN apt-get update && \
    apt-get install -y wget bzip2 git && \
    useradd --create-home \
            --shell /bin/bash \
            --user-group \
            elm && \
    echo 'elm:elm' | chpasswd

# Add files used for development
ADD . /home/elm/elm

# Run chown so we can install
RUN chown -R elm /home/elm/elm

# Login as the elm user
USER elm
WORKDIR /home/elm/

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/elm/miniconda && \
    echo 'export PATH="/home/elm/miniconda/bin:${PATH}"' >> ~/.bashrc
ENV PATH="/home/elm/miniconda/bin:${PATH}"

# Clone down repos
RUN git clone https://github.com/ContinuumIO/earthio.git && \
    git clone https://github.com/ContinuumIO/xarray_filters.git

# Create unified conda environment
RUN conda create -n elm-env -y python=3.5 && \
    conda env update -n elm-env -f earthio/environment.yml
RUN conda env update -n elm-env -f elm/environment.yml

# Install libraries into new environment
RUN cd xarray_filters && ~/miniconda/envs/elm-env/bin/python setup.py develop --no-deps && cd - && \
    cd earthio && ~/miniconda/envs/elm-env/bin/python setup.py develop --no-deps && cd - && \
    cd elm && ~/miniconda/envs/elm-env/bin/python setup.py develop --no-deps && cd -

# Expose port for Jupyter
EXPOSE 8888
