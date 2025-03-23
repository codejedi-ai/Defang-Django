from django.urls import path
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('hello/', views.hello_world, name='hello_world'),
    path('test-style/', views.test_style, name='test_style'),
    path('debug-static/', views.debug_static, name='debug_static'),
    path('login/', views.auth_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
]

# Add this to help with serving static files during development
urlpatterns += staticfiles_urlpatterns()
