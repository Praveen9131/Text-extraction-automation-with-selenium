from django.urls import path
from . import views

## local:8000/sample/index
# sample/urls.py
from django.urls import path
from . import views

# sample/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('store_pdfs/', views.store_pdfs, name='store_pdfs'),
    path('retrieve_answer/', views.retrieve_answer, name='retrieve_answer'),
]

