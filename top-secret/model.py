import numpy as np
import pandas as pd

################################################################
########## Dataset
################################################################

class TopSecretHack():
    """Contains the dataset loading and preprocessing modules.
    """
    def load_datasets(self, file):
        """Loads the main file and assigns the datasets to variables.
        """
        # Main Variables
        assembly_data = pd.read_excel(file, sheet_name = "dataset1")
        initial_data = pd.read_excel(file, sheet_name = "initialinspection")
        final_data = pd.read_excel(file, sheet_name = "finalinspection")
        result_data = pd.read_excel(file, sheet_name = "Final")

        # Extracting specific columns
        aux = initial_data.iloc[:,-3:]
        features = pd.concat([assembly_data.iloc[:,2:], aux_1], axis = 1)
        targets = initial_data.iloc[:,[0,2,8,9,10,11,12]]

        return features, targets

    def one_hotencoding(self, dataset, column):
        """Transforms categorical variables into numerical values.
        """
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        label_encoder = LabelEncoder()
        dataset.iloc[:, column] = label_encoder.fit_transform(dataset.iloc[:,column])
        one_hotencoder = OneHotEncoder(categorical_features = [column])
        dataset = pd.DataFrame(one_hotencoder.fit_transform(dataset).toarray())

        return dataset

    def missing_values(self, dataset, method = "ffill"):
        """Treats missing values of the dataset.
        """
        dataset = dataset.fillna(method = method)

        return dataset

    def scaling_values(self, dataset):
        """Uses z-value to scale values of a dataset.
        """
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        dataset = scaler.fit_transform(dataset)

        return dataset


################################################################
########## Training
################################################################

class Models():

    def train_randomforest(self, features, targets, test_size = 0.25, n_estimators = 10, model_name):
       """Trains a Random Forest Model.
       """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.externals import joblib
        import matplotlib.pyplot as plt

        # Splitting dataset and building the model
        x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size = test_size, random_state = 42)
        model = RandomForestRegressor(n_estimators = 10, random_state = 42, criterion = 'mse',
                                      n_jobs = -1, oob_score = True)
        model.fit(x_train, y_train)

        # Saving the model
        model_name = model_name
        joblib.dump(model, model_name)

        # Testing model and plotting results
        prediction = model.predict(x_test)

        return model

if __name__ == "__main__":
    import argparse

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required = True, help = "path to input dataset")
    ap.add_argument("-m", "--model", required = True, help = "model of choice")
    args = vars(ap.parse_args())

    # Loading features and targets
    dataset = TopSecretHack()
    features, targets = dataset.load_datasets(args["dataset"])

    # One Hot Encoding on categorical variables of targets
    targets = dataset.one_hotencoding(targets, column = -1)

    # Handling Missing Data
    features = dataset.missing_values(features)
    targets = dataset.missing_values(targets)

    if args["model"] == "Random Forest":
        model = Models()
        rf = model.train_randomforest(features, targets, model_name = "RF1.sav")

    