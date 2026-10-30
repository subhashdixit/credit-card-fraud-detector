"""Gunicorn and local Flask entry point."""

from card_fraud.web import create_app

app = create_app()
