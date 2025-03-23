from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
import logging

# Setup logger
logger = logging.getLogger(__name__)

def home(request):
    """
    View function for the home page of the site.
    """
    return render(request, 'home.html')

def logout_view(request):
    """
    View function to handle user logout.
    """
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect('home')

def login_view(request):
    """
    View function for the login page.
    """
    # Get the next URL if provided
    next_url = request.GET.get('next', 'home')
    
    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        remember_me = request.POST.get('remember-me') == 'on'
        
        # Form validation
        if not username:
            messages.error(request, "Username is required.")
            return render(request, 'login.html', {'username': username})
        
        if not password:
            messages.error(request, "Password is required.")
            return render(request, 'login.html', {'username': username})
        
        # Authenticate user
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            # Log successful login attempt
            logger.info(f"Successful login for user: {username}")
            
            # Login the user
            login(request, user)
            
            # Handle remember-me
            if not remember_me:
                request.session.set_expiry(0)  # Session expires when browser closes
            else:
                # Session will expire according to SESSION_COOKIE_AGE in settings
                # Default is 2 weeks if not specified
                pass
                
            # Redirect after login
            return redirect(next_url)
        else:
            # Log failed login attempt
            logger.warning(f"Failed login attempt for username: {username}")
            messages.error(request, "Invalid username or password.")
            return render(request, 'login.html', {'username': username})
            
    return render(request, 'login.html')