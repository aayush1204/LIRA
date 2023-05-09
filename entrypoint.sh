#!/bin/sh

set -e

# python entry point commands go here
python manage.py migrate
python manage.py collectstatic --no-input

exec "$@"