"""
Logging Configuration for TruthScore Application

This module provides centralized logging configuration with multiple log files
for different components and proper log rotation.
"""

import logging
import logging.config
import os
from datetime import datetime

def get_logging_config():
    """
    Returns a comprehensive logging configuration dictionary
    """
    
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(asctime)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'stream': 'ext://sys.stdout'
            },
            'main_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': os.path.join(log_dir, 'truthscore.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf8'
            },
            'analysis_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': os.path.join(log_dir, 'analysis.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf8'
            },
            'api_usage_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': os.path.join(log_dir, 'api_usage.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf8'
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': os.path.join(log_dir, 'errors.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 10,  # Keep more error logs
                'encoding': 'utf8'
            },
            'performance_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': os.path.join(log_dir, 'performance.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf8'
            }
        },
        'loggers': {
            # Main application logger
            'app': {
                'level': 'INFO',
                'handlers': ['console', 'main_file'],
                'propagate': False
            },
            # Analysis-specific logger
            'analysis': {
                'level': 'INFO',
                'handlers': ['analysis_file', 'console'],
                'propagate': False
            },
            # API usage tracking
            'api_usage': {
                'level': 'INFO',
                'handlers': ['api_usage_file'],
                'propagate': False
            },
            # Error tracking
            'errors': {
                'level': 'ERROR',
                'handlers': ['error_file', 'console'],
                'propagate': False
            },
            # Performance monitoring
            'performance': {
                'level': 'INFO',
                'handlers': ['performance_file'],
                'propagate': False
            },
            # Flask and Werkzeug loggers
            'werkzeug': {
                'level': 'WARNING',
                'handlers': ['console', 'main_file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console', 'main_file']
        }
    }
    
    return config

def setup_logging():
    """
    Initialize logging configuration
    """
    config = get_logging_config()
    logging.config.dictConfig(config)
    
    # Log initialization
    logger = logging.getLogger('app')
    logger.info("=" * 80)
    logger.info(f"TruthScore Application Logging Initialized - {datetime.now()}")
    logger.info("=" * 80)
    logger.info("Log files created:")
    logger.info("  - logs/truthscore.log (Main application log)")
    logger.info("  - logs/analysis.log (Analysis workflow)")
    logger.info("  - logs/api_usage.log (OpenAI API usage)")
    logger.info("  - logs/errors.log (Error tracking)")
    logger.info("  - logs/performance.log (Performance monitoring)")
    logger.info("=" * 80)

class PerformanceLogger:
    """
    Context manager for performance logging
    """
    def __init__(self, operation_name, logger_name='performance'):
        self.operation_name = operation_name
        self.logger = logging.getLogger(logger_name)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Started: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            if exc_type:
                self.logger.error(f"Failed: {self.operation_name} (Duration: {duration:.2f}s) - {exc_val}")
            else:
                self.logger.info(f"Completed: {self.operation_name} (Duration: {duration:.2f}s)")

def log_function_call(logger_name='app'):
    """
    Decorator to log function calls with parameters and execution time
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name)
            start_time = datetime.now()
            
            # Log function call (be careful with sensitive data)
            func_name = func.__name__
            logger.debug(f"Calling {func_name}")
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f"{func_name} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"{func_name} failed after {duration:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator 