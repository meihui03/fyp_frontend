# Generated by Django 5.2 on 2025-05-05 09:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0003_users_user'),
    ]

    operations = [
        migrations.CreateModel(
            name='ProtectedUser',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_id', models.CharField(max_length=100, unique=True)),
                ('is_protected', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
    ]
