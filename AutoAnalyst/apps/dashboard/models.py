from django.db import models
from django.contrib.auth.models import User


class Dataset(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE,
        null=True, blank=True, related_name='datasets'
    )
    name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500, blank=True)
    file_size = models.BigIntegerField(default=0)
    rows = models.IntegerField(default=0)
    columns = models.IntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return self.name

    @property
    def size_display(self):
        b = self.file_size
        if b < 1024:
            return f'{b} B'
        elif b < 1024 ** 2:
            return f'{b / 1024:.1f} KB'
        else:
            return f'{b / 1024 ** 2:.1f} MB'


class TrainingRun(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE,
        null=True, blank=True, related_name='training_runs'
    )
    dataset = models.ForeignKey(
        Dataset, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='training_runs'
    )
    dataset_name = models.CharField(max_length=255, blank=True)
    target_column = models.CharField(max_length=255)
    problem_type = models.CharField(max_length=50)
    best_model_name = models.CharField(max_length=255)
    best_score = models.FloatField()
    metric_label = models.CharField(max_length=50)
    n_features = models.IntegerField(default=0)
    n_train = models.IntegerField(default=0)
    n_test = models.IntegerField(default=0)
    all_results = models.JSONField(default=list)
    trained_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-trained_at']

    def __str__(self):
        return f'{self.best_model_name} on {self.dataset_name}'

    @property
    def score_display(self):
        return f'{self.best_score:.1f}%'

    @property
    def problem_type_display(self):
        return self.problem_type.title()


class Prediction(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE,
        null=True, blank=True, related_name='predictions'
    )
    training_run = models.ForeignKey(
        TrainingRun, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='predictions'
    )
    model_name = models.CharField(max_length=255, blank=True)
    target_column = models.CharField(max_length=255, blank=True)
    result_value = models.CharField(max_length=500)
    input_summary = models.JSONField(default=dict)
    predicted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-predicted_at']

    def __str__(self):
        return f'Prediction({self.target_column}={self.result_value})'
