"""chatweb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""

from django.conf.urls import url , include
from django.contrib import admin
from hero.views import video_feed, index, startfunc, recognize, end, start_reg, new, stop_reg, clean

urlpatterns = [
    url(r'^$', index,name='home'),
    url(r'^video_feed/', video_feed, name="video-feed"),
    url(r'^startfunc/', startfunc, name="start"),
    url(r'^recognize/', recognize, name="recognize"),
    url(r'^end/', end, name="end"),
    url(r'^start_reg/', start_reg, name="start_reg"),
    url(r'^new/', new, name="new"),
    url(r'^stop_reg/', stop_reg, name="stop_reg"),
    url(r'^clean/', clean, name="clean"),
]
