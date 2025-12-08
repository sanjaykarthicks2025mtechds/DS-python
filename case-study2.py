

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

digits=load_digits()
x=pd.DataFrame(digits.data,columns=digits.feature_names)
y=pd.Series(digits.target,name="target")

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
activations=["tanh","relu","logistic"]
results={}

for act in activations:
    print(act)
    mlp=MLPClassifier(
        activation=act,
        hidden_layer_sizes=(128,64),
        solver="adam",
        batch_size=64,
        max_iter=30,
        random_state=42,
        early_stopping=True,
        verbose=False,
    )
    mlp.fit(x_train,y_train)
    y_pred=mlp.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    results[act]=acc

    print(f"Accuracy ({act}): {acc:.4f}")

print("\n=== Activation comparison ===")
for act, acc in results.items():
    print(f"{act:10s} -> {acc:.4f}")