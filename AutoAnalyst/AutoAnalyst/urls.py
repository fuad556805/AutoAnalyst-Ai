from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from apps.core import views as core_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', core_views.home, name='home'),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('signup/', core_views.signup, name='signup'),
    path('dashboard/', include('apps.dashboard.urls')),
    path('dataset/', include('apps.dataset.urls')),
    path('ml/', include('apps.ml.urls')),
    path('predict/', include('apps.predict.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
