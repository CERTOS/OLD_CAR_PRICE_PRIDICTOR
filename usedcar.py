
import numpy as np 
import pandas as pd 
import joblib
# Read the CSV file 
rider_provider='CARS.csv'
car_dataframe = pd.read_csv(rider_provider)
#thus there are two null enteries in the Cylinder feature of the car, so lets remove it
car_dataframe = car_dataframe.dropna()
#removing currency sign and other unrequired signs from price related columms and converting it to integer form
car_dataframe["MSRP"] = car_dataframe["MSRP"].str.replace("$", "")
car_dataframe["MSRP"] = car_dataframe["MSRP"].str.replace(",", "")
car_dataframe["MSRP"] = car_dataframe["MSRP"].astype(int)
car_dataframe["Invoice"] = car_dataframe["Invoice"].str.replace("$", "")
car_dataframe["Invoice"] = car_dataframe["Invoice"].str.replace(",", "")
car_dataframe["Invoice"] = car_dataframe["Invoice"].astype(int)
car_dataframe=car_dataframe.drop(['Model'], axis=1)
# Perform One-Hot Encoding for "Make", "Type", "Origin", and "DriveTrain"
df_dum = pd.get_dummies(car_dataframe, columns = ['Make','Type','Origin','DriveTrain'])
#DROPPING A COLUMN TO AVOID HOT TRAP.
dataframe_data = df_dum.drop(['Invoice'], axis=1)
##lets check the shape of our new dataframe
dataframe_data.shape
# Feeding input features to X and output (MSRP) to y
X = dataframe_data.drop("MSRP", axis = 1)
Y = dataframe_data["MSRP"]
modelcol=["EngineSize","Cylinders","Horsepower","MPG_City","MPG_Highway","Weight","Wheelbase","Length","Make_Acura","Make_Audi","Make_BMW","Make_Buick","Make_Cadillac","Make_Chevrolet","Make_Chrysler","Make_Dodge","Make_Ford","Make_GMC","Make_Honda","Make_Hummer","Make_Hyundai","Make_Infiniti","Make_Isuzu","Make_Jaguar","Make_Jeep","Make_Kia","Make_Land Rover","Make_Lexus","Make_Lincoln","Make_MINI","Make_Mazda","Make_Mercedes-Benz","Make_Mercury","Make_Mitsubishi","Make_Nissan","Make_Oldsmobile","Make_Pontiac","Make_Porsche","Make_Saab","Make_Saturn","Make_Scion","Make_Subaru","Make_Suzuki","Make_Toyota","Make_Volkswagen","Make_Volvo","Type_Hybrid","Type_SUV","Type_Sedan","Type_Sports","Type_Truck","Type_Wagon","Origin_Asia","Origin_Europe","Origin_USA","DriveTrain_All","DriveTrain_Front","DriveTrain_Rear",]
joblib.dump(modelcol,"modelcol.pkl")
## creating X and Y numpy araay to make it ready for dataset splitiing for test and train
X = np.array(X)
Y = np.array(Y)
#spliting data i am using test train split from sklearn, thus importing it
from sklearn.model_selection import train_test_split
# spliting the dataset into 80-20 % partition
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size = 0.2)
##lets name our model as : DEVIL'S_EYE_MODEL
from xgboost import XGBRegressor
DEVIL_EYE_MODEL = XGBRegressor()
DEVIL_EYE_MODEL.fit(X_train, Y_train)
accuracy_of_DEVIL_EYE_MODEL = DEVIL_EYE_MODEL.score(X_test, Y_test)
percentage_accuracy_of_DEVIL_EYE_MODEL = accuracy_of_DEVIL_EYE_MODEL * 100
print("Accuracy of DEVIL_EYE_MODEL i.e. Random forest model with XGBOOST is : ",percentage_accuracy_of_DEVIL_EYE_MODEL)
import joblib
joblib.dump(DEVIL_EYE_MODEL, 'model.pkl')

'''  maker= request.form["maker"]
        Type = request.form["type"]
        origin = request.form["origin"]
        drivetrain = request.form["drivetrain"]
        EngineSize = request.form["EngineSize"]
        Cylinders = request.form["Cylinders"]
        Horsepower = request.form["Horsepower"]
        MPG_City = request.form["MPG_City"]
        MPG_Highway = request.form["MPG_Highway"]
        Weight = request.form["Weight"]
        Wheelbase = request.form["Wheelbase"]
        Length = request.form["Length"]
'''