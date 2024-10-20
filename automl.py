from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo 

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes to numpy arrays) 
X = breast_cancer_wisconsin_diagnostic.data.features.values
y = breast_cancer_wisconsin_diagnostic.data.targets.values
  
# Encode y
y = y.flatten()
le = LabelEncoder()
y = le.fit_transform(y)

# variable information 
print(breast_cancer_wisconsin_diagnostic.variables) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

pipeline = TPOTClassifier(generations=10,
                          population_size=20,
                          cv=5,
                          random_state=42,
                          n_jobs=-1)

pipeline.fit(X_train, y_train)
print(pipeline.score(X_test, y_test))
pipeline.export('BreastCancerAutoML.py')