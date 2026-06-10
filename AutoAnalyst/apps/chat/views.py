import json
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib import messages

from .models import ChatMessage
from .gemini_client import (
    load_dataset, build_dataset_context, build_system_prompt,
    parse_response, generate_chart, chat_with_gemini,
)


@login_required
def chat_view(request):
    msgs         = ChatMessage.objects.filter(user=request.user).order_by('created_at')
    dataset_name = request.session.get('dataset_name', '')
    has_dataset  = bool(request.session.get('dataset_path'))
    has_model    = bool(request.session.get('model_dir'))
    best_model   = request.session.get('best_model_name', '')
    problem_type = request.session.get('problem_type', '')
    best_score   = ''
    metric_label = request.session.get('ml_metric_label', '')
    ml_results   = request.session.get('ml_results', [])
    for r in ml_results:
        if r.get('name') == best_model:
            best_score = r.get('score', '')
            break

    return render(request, 'chat.html', {
        'chat_messages': msgs,
        'dataset_name':  dataset_name,
        'has_dataset':   has_dataset,
        'has_model':     has_model,
        'best_model':    best_model,
        'problem_type':  problem_type,
        'best_score':    best_score,
        'metric_label':  metric_label,
    })


@login_required
@require_POST
def send_message(request):
    try:
        body         = json.loads(request.body)
        user_message = body.get('message', '').strip()
    except Exception:
        user_message = request.POST.get('message', '').strip()

    if not user_message:
        return JsonResponse({'error': 'Empty message'}, status=400)

    dataset_path = request.session.get('dataset_path')
    dataset_name = request.session.get('dataset_name', 'dataset')
    df, _        = load_dataset(dataset_path)
    ctx          = build_dataset_context(df) if df is not None else None

    ml_context = None
    if request.session.get('model_dir'):
        best_model   = request.session.get('best_model_name', '')
        ml_results   = request.session.get('ml_results', [])
        best_score   = next((r['score'] for r in ml_results if r.get('name') == best_model), '')
        ml_context   = {
            'best_model_name': best_model,
            'problem_type':    request.session.get('problem_type', ''),
            'best_score':      best_score,
            'metric_label':    request.session.get('ml_metric_label', ''),
        }

    history_qs = ChatMessage.objects.filter(user=request.user).order_by('-created_at')[:20]
    history    = [{'role': m.role, 'content': m.content} for m in reversed(history_qs)]

    system_prompt = build_system_prompt(ctx, dataset_name, ml_context)
    raw_text, error = chat_with_gemini(user_message, system_prompt, history)

    ChatMessage.objects.create(user=request.user, role='user', content=user_message)

    if error:
        msg = ChatMessage.objects.create(user=request.user, role='assistant', content=error)
        return JsonResponse({'text': error, 'chart': None, 'pdf_requested': False, 'msg_id': msg.pk})

    clean_text, chart_spec, pdf_requested = parse_response(raw_text)

    chart_b64 = None
    if chart_spec and df is not None:
        chart_b64 = generate_chart(df, chart_spec['type'], chart_spec['col1'], chart_spec.get('col2'))

    msg = ChatMessage.objects.create(
        user=request.user,
        role='assistant',
        content=clean_text,
        chart_data=chart_b64 or '',
    )
    return JsonResponse({
        'text':          clean_text,
        'chart':         chart_b64,
        'pdf_requested': pdf_requested,
        'msg_id':        msg.pk,
    })


@login_required
@require_POST
def clear_history(request):
    ChatMessage.objects.filter(user=request.user).delete()
    return JsonResponse({'status': 'ok'})


@login_required
def chat_pdf(request):
    if not request.session.get('model_dir'):
        messages.error(request, 'No trained model found. Please train a model first.')
        return redirect('chat')
    return redirect('report_pdf')
