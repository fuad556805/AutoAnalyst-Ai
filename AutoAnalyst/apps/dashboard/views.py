import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.contrib import messages
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


@login_required(login_url='/login/')
def load_dataset(request, pk):
    ds = get_object_or_404(Dataset, pk=pk, user=request.user)
    if not ds.file_path or not os.path.exists(ds.file_path):
        messages.error(request, 'Dataset file no longer exists on disk.')
        return redirect('dashboard')
    request.session['dataset_path']       = ds.file_path
    request.session['dataset_name']       = ds.name
    request.session['current_dataset_id'] = ds.pk
    return redirect('preview')


@login_required(login_url='/login/')
@require_POST
def delete_dataset(request, pk):
    ds = get_object_or_404(Dataset, pk=pk, user=request.user)
    ds.delete()
    messages.success(request, f'Dataset "{ds.name}" deleted.')
    return redirect('dashboard')
