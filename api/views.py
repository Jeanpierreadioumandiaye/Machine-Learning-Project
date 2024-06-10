from django.shortcuts import render
from django.http import JsonResponse
import pickle
import pandas as pd




# Home page
def home(request):
    
    return render(request, 'user.html')

def churn_prediction(request):
    if request.method == 'POST':
        data = {
        'CreditScore': [request.POST.get('credit_score')],
        'Geography': [request.POST.get('geography')],
        'Gender': [request.POST.get('gender')],
        'Age': [request.POST.get('age')],
        'Tenure': [request.POST.get('tenure')],
        'Balance': [request.POST.get('balance')],
        'NumOfProducts': [request.POST.get('num_of_products')],
        'HasCrCard': [request.POST.get('has_cr_card')],
        'IsActiveMember': [request.POST.get('is_active_member')],
        'EstimatedSalary': [request.POST.get('estimated_salary')]
    }

        
        # Convert the data to a DataFrame
        df = pd.DataFrame(data)
        
        
        # Load the model
        with open('static/model/best_xgb.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Make the prediction
        y_predict = model.predict(df)
        
        return JsonResponse({'prediction': float(y_predict)})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
