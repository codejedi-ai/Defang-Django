from django.shortcuts import render
from .models import Greeting

def index(request):
    # Create a default greeting if none exists
    greeting, created = Greeting.objects.get_or_create(pk=1)
    
    context = {
        'greeting': greeting.message
    }
    return render(request, 'index.html', context)
