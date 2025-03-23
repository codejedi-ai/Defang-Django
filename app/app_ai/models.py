from django.db import models

# Simple model for Hello World site
class Greeting(models.Model):
    message = models.CharField(max_length=100, default="Hello, World!")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.message
