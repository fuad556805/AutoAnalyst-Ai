from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Dataset, TrainingRun, Prediction


@login_required(login_url='/login/')
def dashboard(request):
    user = request.user

    datasets = Dataset.objects.filter(user=user).order_by('-uploaded_at')
    training_runs = TrainingRun.objects.filter(user=user).order_by('-trained_at')
    predictions = Prediction.objects.filter(user=user).order_by('-predicted_at')

    total_datasets = datasets.count()
    total_training_runs = training_runs.count()
    total_predictions = predictions.count()

    best_score = None
    best_model = None
    best_problem_type = None
    if training_runs.exists():
        best_run = training_runs.order_by('-best_score').first()
        best_score = best_run.best_score
        best_model = best_run.best_model_name
        best_problem_type = best_run.problem_type

    classification_count = training_runs.filter(problem_type='classification').count()
    regression_count = training_runs.filter(problem_type='regression').count()

    recent_datasets = datasets[:5]
    recent_training_runs = training_runs[:10]
    recent_predictions = predictions[:15]

    context = {
        'total_datasets': total_datasets,
        'total_training_runs': total_training_runs,
        'total_predictions': total_predictions,
        'best_score': best_score,
        'best_model': best_model,
        'best_problem_type': best_problem_type,
        'classification_count': classification_count,
        'regression_count': regression_count,
        'recent_datasets': recent_datasets,
        'all_datasets': datasets,
        'recent_training_runs': recent_training_runs,
        'recent_predictions': recent_predictions,
    }
    return render(request, 'dashboard.html', context)
