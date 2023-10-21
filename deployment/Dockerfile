FROM python:3.9

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

WORKDIR /code

COPY ["Pipfile", "Pipfile.lock", "./code/"]

RUN cd code && pip install pipenv && pipenv install --system --deploy --ignore-pipfile

COPY ["model.py", "app.py", "/code/"]
COPY ["models", "/code/models/"]
COPY ["data", "/code/data/"]

CMD ["gradio", "/code/app.py"]