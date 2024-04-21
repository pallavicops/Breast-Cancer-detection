from flask import Flask, request, render_template
import pickle   # for loading the model
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/",methods=['GET'])
def home():
  return render_template('index.html')

@app.route('/form',methods=['GET'])
def form():
  return render_template('form.html')

@app.route('/predict',methods=['POST'])
def predict():
 input_features = [float(x) for x in request.form.values()]
 value= [np.array(input_features)]
 print(value)
 
 output = model.predict(value)
 if output == 1:
  return render_template('yes.html')

 else:
  return render_template('no.html')


if __name__ == '__main__':
    app.run(port=3000, debug=True)

    # features_value = [np.array([-0.23927557,  1.20953909, -0.30776593, -0.32756529, -0.57492927,
  #       -0.92580153, -0.73956439, -0.71148962, -0.80536136, -0.9836759 ,
  #       -0.80543757,  0.21044227, -0.85774844, -0.62021463, -0.41371289,
  #       -0.91150812, -0.66647515, -0.81890411, -0.87464651, -0.82601506,
  #       -0.38282965,  1.32199868, -0.47336605, -0.44009447, -0.13610472,
  #       -0.88286449, -0.59768168, -0.79588429, -0.81775175])]
  # 'radius_mean', 'texture_mean', 'perimeter_mean',
  #      'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
  #      'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
  #      'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
  #      'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
  #      'fractal_dimension_se', 'radius_worst', 'texture_worst',
  #      'perimeter_worst', 'area_worst', 'smoothness_worst',
  #      'compactness_worst', 'concavity_worst', 'concave points_worst',
  #      'symmetry_worst', ],
  #     )



#   df = pd.DataFrame(features_value, columns=features_name)
