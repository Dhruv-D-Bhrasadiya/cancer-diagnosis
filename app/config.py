# Configuration for Flask Application

import os

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # File upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'txt'}
    
    # Model settings
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    
    # Feature importance settings
    TOP_K_FEATURES = 20


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'change-this-in-production'
    
    @classmethod
    def validate(cls):
        """Validate production config"""
        if not os.environ.get('SECRET_KEY'):
            raise ValueError("SECRET_KEY environment variable must be set in production")


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True


def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    
    if env == 'production':
        config = ProductionConfig()
        config.validate()
        return config
    elif env == 'testing':
        return TestingConfig()
    else:
        return DevelopmentConfig()
