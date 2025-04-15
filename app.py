from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load all artifacts
artifacts = joblib.load('project_classifier2.pkl')  # Should contain: model, tfidf, panel_encoder, mlb

@app.route('/')
def home():
    return render_template('index.html', 
                         prediction=None,
                         research_areas=None,
                         title='',
                         keywords='',
                         area='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        title = request.form['title']
        keywords = request.form['keywords']
        
        # Create combined text (same format as training)
        combined_text = f"{title} {keywords}"
        
        # Transform text
        X = artifacts['tfidf'].transform([combined_text])
        
        # Make prediction
        y_pred = artifacts['model'].predict(X)
        panel = artifacts['panel_encoder'].inverse_transform([y_pred[0, 0]])[0]
        research_areas = artifacts['mlb'].inverse_transform(y_pred[:, 1:])[0]
        
        return render_template('index.html', 
                            prediction=panel,
                            research_areas=", ".join(research_areas),
                            title=title,
                            keywords=keywords,
                            area='')  # Clear research areas input since we predict it
        
    except Exception as e:
        return render_template('index.html', 
                            prediction="Error",
                            research_areas=None,
                            error=str(e),
                            title=request.form.get('title', ''),
                            keywords=request.form.get('keywords', ''),
                            area=request.form.get('area', ''))

if __name__ == '__main__':
    app.run(debug=True)