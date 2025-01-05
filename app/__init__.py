import os
from flask import Flask

from .routes.main_routes import main_bp
from .routes.inference_routes import inference_bp
from .config import get_config 

def create_app():
    app = Flask(__name__, instance_relative_config=True)

    config_obj = get_config()

    for key in dir(config_obj):
        if key.isupper():
            app.config[key] = getattr(config_obj, key)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    app.register_blueprint(main_bp)
    app.register_blueprint(inference_bp, url_prefix="/api")

    return app
