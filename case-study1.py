import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def inject_missing(df,frac=0.05):
    df_missing=df.copy()
    mask=np.random.rand(*df_missing.shape)<frac
    df_missing=df_missing.mask(mask)
    return df_missing

def outlier(df,feature_cols,k=1.5):
    df_clean=df.copy()
    for col in feature_cols:
        q1=df_clean[col].quantile(0.25)
        q3=df_clean[col].quantile(0.75)
        iqr=q3-q1
        lower=q1 - k*iqr
        upper=q3 + k*iqr
        df_clean=df_clean[(df_clean[col]>=lower) & (df_clean[col]<=upper)]
    return df_clean

cancer=load_breast_cancer()
x=pd.DataFrame(cancer.data,columns=cancer.feature_names)
y=pd.Series(cancer.target,name="target")



print("Original clf shape:", x.shape)
x = inject_missing(x, frac=0.05)
print("With missing values:", x.isna().sum().sum(), "missing")

df=x.copy()
df["target"]=y

feature_cols=cancer.feature_names
df=outlier(df, feature_cols, k=1.5)

x=df[feature_cols]
y=df["target"]


pipl=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler()),
    ("model",RandomForestClassifier(
        n_estimators=300,
        random_state=42,
    )),
])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42,stratify=y)

pipl.fit(x_train,y_train)
y_pred=pipl.predict(x_test)

print("\n=== CLASSIFICATION RESULTS ===")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))

imps=pipl.named_steps["model"].feature_importances_
feat_imp=pd.Series(imps,index=feature_cols).sort_values(ascending=False)

print("\nTop 10 important features (classifier):")
print(feat_imp.head(10))
