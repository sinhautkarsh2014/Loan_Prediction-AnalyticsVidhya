import pandas as pd 
import pylab
df=pd.read_csv("creditrain.csv")


df["Gender"].fillna(value="other", inplace=True)
df["Genderc"]=df["Gender"].astype("category")
df["Genderc"]=df["Genderc"].cat.rename_categories([1,2,3])
df=df.drop("Gender", axis=1)


df["Married"].fillna(value="No", inplace=True)
df["Marriedc"]=df["Married"].astype("category")
df["Marriedc"]=df["Marriedc"].cat.rename_categories([1,2])
df=df.drop("Married", axis=1)


df["Dependents"].fillna(value="0", inplace=True)
df["Dependentsc"]=df["Dependents"].astype("category")
df["Dependentsc"]=df["Dependentsc"].cat.rename_categories([1,2,3,4])
df=df.drop("Dependents", axis=1)



df["Educationc"]=df["Education"].astype("category")
df["Educationc"]=df["Educationc"].cat.rename_categories([1,2])
df=df.drop("Education", axis=1)



df["Self_Employed"].fillna(value="other", inplace=True)
df["Self_Employedc"]=df["Self_Employed"].astype("category")
df["Self_Employedc"]=df["Self_Employedc"].cat.rename_categories([1,2,3])
df=df.drop("Self_Employed", axis=1)

df["Property_Areac"]=df["Property_Area"].astype("category")
df["Property_Areac"]=df["Property_Areac"].cat.rename_categories([1,2,3])
df=df.drop("Property_Area", axis=1)


df["Credit_History"].fillna(value=1, inplace=True)
df["Credit_Historyc"]=df["Credit_History"].astype("category")
df=df.drop("Credit_History", axis=1)


df["ApplicantIncome"]=df["ApplicantIncome"].astype("object")
df["CoapplicantIncome"]=df["CoapplicantIncome"].astype("object")


df["LoanAmount"]=df["LoanAmount"].fillna(df["LoanAmount"].mean())
df["LoanAmount"]=df["LoanAmount"].astype("object")


df["Loan_Amount_Term"]=df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mean())
df["Loan_Amount_Term"]=df["Loan_Amount_Term"].astype("object")

from sklearn.preprocessing import scale
e=pd.Series(data=10.5, index=range(0,981))
a = df["LoanAmount"].multiply(9.5*1000)
b=a.multiply(e.pow((df["Loan_Amount_Term"].divide(12))))
d=e.pow((df["Loan_Amount_Term"].divide(12).subtract(1)))
EMI=b.divide(d)

df["sum_Income"]=df["ApplicantIncome"].add(df["CoapplicantIncome"])
df["Poly_feature"]=EMI.divide((df["sum_Income"]))
df["ratio"]=df["sum_Income"].divide(df["LoanAmount"])

df["Poly_feature"]=scale(df["Poly_feature"])
df["sum_Income"]=scale(df["sum_Income"])
df["LoanAmount"]=scale(df["LoanAmount"])
df["Loan_Amount_Term"]=scale(df["Loan_Amount_Term"])
df["ratio"]=scale(df["ratio"])

df_train=df[df.Loan_Status.notnull()]
df_test=df[df.Loan_Status.isnull()]

df_test=df_test.drop("Loan_Status", axis=1)
key=df_test["Loan_ID"]
df_test=df_test.drop("Loan_ID", axis=1)
df_train=df_train.drop("Loan_ID", axis=1)

from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
Xtrsp=pd.DataFrame(enc.fit_transform(df_train[["Genderc","Marriedc","Dependentsc","Educationc","Self_Employedc","Property_Areac", "Credit_Historyc"]]).toarray())
Xtstsp=pd.DataFrame(enc.transform(df_test[["Genderc","Marriedc","Dependentsc","Educationc","Self_Employedc","Property_Areac", "Credit_Historyc"]]).toarray())
print Xtrsp.shape

df_train["Loan_Statusc"]=df_train["Loan_Status"].astype("category")
df_train["Loan_Statusc"]=df_train["Loan_Statusc"].cat.rename_categories([1,0])


y_train=df_train["Loan_Statusc"]
df_train=df_train.drop("Loan_Status", axis=1)
df_train=df_train.drop("Loan_Statusc", axis=1)




from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
sel = SelectKBest(chi2, k=10)
Xtrsp=pd.DataFrame(sel.fit_transform(Xtrsp,y_train))
Xtstsp=pd.DataFrame(sel.transform(Xtstsp))



Xtrf=df_train[["Poly_feature","sum_Income","LoanAmount","Loan_Amount_Term","ratio"]]
Xtstf=df_test[["Poly_feature","sum_Income","LoanAmount","Loan_Amount_Term","ratio"]]

Xtrf.index=Xtrsp.index
X_train=pd.concat([Xtrsp,Xtrf], axis=1)
Xtstf.index=Xtstsp.index
X_test=pd.concat([Xtstsp,Xtstf], axis=1)

from sklearn.linear_model import LogisticRegression 
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier 
#from sklearn.naive_bayes import GaussianNB
#from sklearn.neural_network import MLPClassifier
#from sklearn.svm import LinearSVC

#model=RandomForestClassifier(n_estimators=128, verbose=1)
model=LogisticRegression(C=0.01, tol=0.00001)
#model=KNeighborsClassifier(n_neighbors=40)
#model=LinearSVC(C=0.01, tol=0.0000001)
#model=GaussianNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print y_pred

out=pd.DataFrame()
out["Loan_ID"]=key
out["Loan_Status"]=y_pred
out["Loan_Status"]=out["Loan_Status"].astype("category")
out["Loan_Status"]=out["Loan_Status"].cat.rename_categories(["Y","N"])
out.to_csv('outcredit.csv', index=False)






#from sklearn.cross_validation import train_test_split

#Xtr,Xts,ytr,yts=train_test_split(X_train,y_train, test_size=0.1)
#model.fit(Xtr, ytr)
#print model.score(Xts,yts)










