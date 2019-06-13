FROM nvcr.io/nvidia/pytorch:19.01-py3

# run this before copying requirements for cache efficiency
RUN pip install --upgrade pip

# working directory
WORKDIR /flask-app

# install in temp directory
RUN mkdir /dependencies

COPY requirements.txt /dependencies/requirements.txt
RUN cd /dependencies && pip3 install -r requirements.txt

# copy code itself from context to image
COPY . /flask-app

# copy generated dependencies
RUN cp -r /dependencies/* /flask-app/

CMD ["gunicorn", "--workers=4", "-b 0.0.0.0:5000","wsgi:app"]

EXPOSE 5000
