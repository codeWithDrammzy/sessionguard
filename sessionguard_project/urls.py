"""
URL configuration for sessionguard_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from core.views import SessionEventView

urlpatterns = [
    path('admin/', admin.site.urls),
    # One view class, channel bound per-URL (see core/views.py docstring).
    path('api/session-event/',
         SessionEventView.as_view(channel='app'), name='session-event'),
    path('api/ussd-event/',
         SessionEventView.as_view(channel='ussd'), name='ussd-event'),
]
