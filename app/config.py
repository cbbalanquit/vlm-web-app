"""Module for application configuration"""
import os
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
ENV_FILE = os.path.join(PROJECT_ROOT, '.env')

if os.path.exists(ENV_FILE):
    load_dotenv(ENV_FILE)
else:
    os.environ.setdefault('FLASK_ENV', 'development')

class Config:
    """Base configuration class"""
    def __init__(self):
        self.FLASK_ENV = os.getenv('FLASK_ENV', 'development')

        self.SQLITE_DB_PATH = os.path.join(
            PROJECT_ROOT,
            'instance',
            'auth.db'
        )

        self.SECRET_KEY = os.getenv('SECRET_KEY', 'not-so-secret')

class DevelopmentConfig(Config):
    """Development-specific configuration"""
    def __init__(self):
        super().__init__()
        self.DEBUG = True
        self.TESTING = False

class ProductionConfig(Config):
    """Production-specific configuration"""
    def __init__(self):
        super().__init__()
        self.DEBUG = False
        self.TESTING = False

        if self.SECRET_KEY == 'not-so-secret':
            raise ValueError("Production SECRET_KEY must be set in environment")

def get_config():
    """
    Dynamically select configuration based on FLASK_ENV.
    """
    env = os.getenv('FLASK_ENV', 'development')
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig
    }
    return configs.get(env, DevelopmentConfig)()
