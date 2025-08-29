from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AggregationViewSet

router = DefaultRouter()
router.register(r'aggregation', AggregationViewSet, basename='aggregation')

urlpatterns = [
    path('', include(router.urls)),
] 