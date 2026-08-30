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

from core.bank_views import (
    BankAppView, bank_login, bank_lookup_account, bank_send_money,
    bank_set_pin, bank_signup, bank_state, bank_verify_pin,
)
from core.demo_views import ControlRoomView, demo_scenarios, toggle_offline
from core.views import SessionEventView

urlpatterns = [
    path('admin/', admin.site.urls),
    # One view class, channel bound per-URL (see core/views.py docstring).
    path('api/session-event/',
         SessionEventView.as_view(channel='app'), name='session-event'),
    path('api/ussd-event/',
         SessionEventView.as_view(channel='ussd'), name='ussd-event'),
    # --- Demo-only routes (presentations; not for production) -----------
    path('api/demo/scenarios/', demo_scenarios, name='demo-scenarios'),
    path('api/demo/toggle-offline/', toggle_offline,
         name='demo-toggle-offline'),
    path('demo/', ControlRoomView.as_view(), name='control-room'),
    # --- Demo customer experience (bank app + USSD simulator) -----------
    path('bank/', BankAppView.as_view(), name='bank-app'),
    path('api/bank/signup/', bank_signup, name='bank-signup'),
    path('api/bank/login/', bank_login, name='bank-login'),
    path('api/bank/set-pin/', bank_set_pin, name='bank-set-pin'),
    path('api/bank/verify-pin/', bank_verify_pin, name='bank-verify-pin'),
    path('api/bank/state/<uuid:user_id>/', bank_state, name='bank-state'),
    path('api/bank/send-money/', bank_send_money, name='bank-send-money'),
    path('api/bank/lookup-account/', bank_lookup_account, name='bank-lookup-account'),
]
