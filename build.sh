#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

cd AutoAnalyst
python manage.py collectstatic --noinput
python manage.py migrate
python manage.py shell -c "
import os
from django.contrib.auth import get_user_model
User = get_user_model()
username = os.environ.get('DJANGO_SUPERUSER_USERNAME', 'admin')
email = os.environ.get('DJANGO_SUPERUSER_EMAIL', 'admin@example.com')
password = os.environ.get('DJANGO_SUPERUSER_PASSWORD', '')
if not password:
    print('DJANGO_SUPERUSER_PASSWORD not set — skipping.')
elif User.objects.filter(username=username).exists():
    print(f'Superuser [{username}] already exists — skipping.')
else:
    User.objects.create_superuser(username=username, email=email, password=password)
    print(f'Superuser [{username}] created successfully.')
"
