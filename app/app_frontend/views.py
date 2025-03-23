from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.views.decorators.csrf import ensure_csrf_cookie
from app_frontend.forms import LoginForm, SignupForm
import os

def home(request):
    """Home page view"""
    return render(request, os.path.join('frontend', 'home.html'))

def hello_world(request):
    """Simple view that returns a greeting"""
    return JsonResponse({'message': 'Hello, World!'})

def test_style(request):
    """View to test if CSS styles are loading properly"""
    return render(request, 'test_style.html')

def debug_static(request):
    """View for debugging static files"""
    static_dirs = settings.STATICFILES_DIRS
    static_url = settings.STATIC_URL
    
    # Check if styles.css exists in the static directory
    css_exists = False
    css_path = None
    
    for static_dir in static_dirs:
        path = os.path.join(static_dir, 'css', 'styles.css')
        if os.path.exists(path):
            css_exists = True
            css_path = path
            break
    
    debug_info = {
        'static_dirs': static_dirs,
        'static_url': static_url,
        'css_exists': css_exists,
        'css_path': css_path,
    }
    
    return render(request, 'debug_static.html', {'debug_info': debug_info})

@ensure_csrf_cookie
def auth_view(request):
    """
    View to handle authentication (login/signup).
    Uses @ensure_csrf_cookie to make sure the CSRF token is set in the response.
    """
    login_form = LoginForm()
    signup_form = SignupForm()
    
    if request.method == 'POST':
        mode = request.POST.get('mode')
        
        if mode == 'login':
            # Handle login with form
            login_form = LoginForm(data=request.POST)
            if login_form.is_valid():
                user = login_form.get_user()
                if user is not None:
                    # Set session expiry based on remember me checkbox
                    if login_form.cleaned_data.get('remember_me'):
                        request.session.set_expiry(1209600)  # 2 weeks
                    else:
                        request.session.set_expiry(0)  # Until browser is closed
                        
                    login(request, user)
                    messages.success(request, f'Welcome back, {user.email}!')
                    return redirect('home')
            
            # Form is invalid, show error
            return render(request, 'frontend/auth.html', {
                'login_form': login_form,
                'signup_form': signup_form,
                'active_form': 'login',
                'error': 'Invalid credentials'
            })
                
        elif mode == 'signup':
            # Handle signup with form
            signup_form = SignupForm(request.POST)
            if signup_form.is_valid():
                user = signup_form.save()
                login(request, user)
                messages.success(request, f'Account created successfully! Welcome, {user.email}!')
                return redirect('home')
            
            # Form is invalid, show error
            return render(request, 'frontend/auth.html', {
                'login_form': login_form,
                'signup_form': signup_form,
                'active_form': 'signup'
            })
    
    # GET request - just display the form
    return render(request, 'frontend/auth.html', {
        'login_form': login_form,
        'signup_form': signup_form,
        'active_form': 'login'
    })

def logout_view(request):
    """Logout view"""
    logout(request)
    messages.info(request, 'You have been logged out successfully.')
    return redirect('home')