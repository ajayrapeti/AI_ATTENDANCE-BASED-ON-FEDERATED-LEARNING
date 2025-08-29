from django.db import models

# Create your models here.

class Student(models.Model):
    student_id = models.CharField(max_length=50, unique=True)
    embedding_file = models.FileField(upload_to='embeddings')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Student {self.student_id}"

class GlobalModel(models.Model):
    version = models.CharField(max_length=50)
    model_file = models.FileField(upload_to='global_model')
    num_students = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Model v{self.version} ({self.num_students} students)"

class TestImage(models.Model):
    image = models.ImageField(upload_to='test_images')
    result_image = models.ImageField(upload_to='test_results', null=True, blank=True)
    processed = models.BooleanField(default=False)
    result = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Test Image {self.id}"
