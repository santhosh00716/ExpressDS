"""
Data Engine - Handles file uploads and automated data cleaning
"""
from .cleaner import clean_data, get_data_health_report

__all__ = ["clean_data", "get_data_health_report"]
