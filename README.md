# VLM Demo
This project is a demo of the VLM (Video Language Model) model. The VLM model is a multimodal model that combines video and text data to generate image or video captions.

Setup guide for running in:
- [Usual Windows, Mac, or Linux with GPU](#usual-pc-with-gpu)
- [Edge Device like Jetson Nano using ARM architecture](#edge-device-like-jetson-nano)

Note: If you wish to change the model used, you can:
- Change the models usied in the existing services in `app/services` folder or create a new service with the desired model
- Change the inference route in `app/routes/inference_routes.py` to use the modified or new service

## Usual PC with GPU
1. Clone the repository
2. Ensure to add environment variable `GPU_INDEX=<desired_index>` to use the appropriate GPU. Default is 0.
3. Setup the application environment either using `requirements.txt` or the [uv package manager](https://docs.astral.sh/uv/) then run the application:
    - Using `requirements.txt`:
        - Create a virtual environment using `python -m venv .venv`
        - Activate the virtual environment using `source .venv/bin/activate`
        - Install the requirements using `pip install -r requirements.txt`
        - Run the application using `python run.py`
    - Using `uv` package manager:
        - Install `uv` package manager going to the [official installation page](https://docs.astral.sh/uv/getting-started/installation/)
        - Run `uv run run.py`
4. The application will start. You can access the application using the url `http://localhost:5000`
5. Go to the application url using your browser and allow the application to access your camera

## Edge Device like Jetson Nano
1. Setup the microSD card with the Jetson Nano image using the [official guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)
2. Clone the repository
3. Create a virtual environment using `python -m venv .venv`
4. Activate the virtual environment using `source .venv/bin/activate`
5. Install the following dependencies:
    - flask: `pip install flask`
    - numpy version 1.24.1 or lower: `pip install numpy==1.24.1`
    - pillow: `pip install Pillow`
    - transformers: `pip install transformers`
    - python-dotenv: `pip install python-dotenv`
    - torch: Follow the instructions in the [official Nvidia website](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html#overview__section_orin)
        - Note: make sure to use appropriate wheel file for your Jetson Nano version which aligns with the jetpack version and the python version you are using. For me, I used this [wheel file](https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl)
6. Run the application using `python run.py`