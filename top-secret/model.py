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
        features = pd.concat([assembly_data.iloc[:,2:], aux], axis = 1)
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

    def train_randomforest(self, features, targets, model_name, test_size = 0.25, n_estimators = 10):
        """Trains a Random Forest Model.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        from sklearn.externals import joblib
        import matplotlib.pyplot as plt

        # Splitting dataset and building the model
        x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size = test_size, random_state = 42)
        model = RandomForestRegressor(n_estimators = 100, random_state = 42, criterion = 'mse',
                                      n_jobs = -1, oob_score = True)
        model.fit(x_train, y_train)

        # Saving the model
        model_name = model_name
        joblib.dump(model, model_name)

        # Testing model and plotting results
        prediction = model.predict(x_test)
        print(prediction)
        r2 = r2_score(y_test, prediction)

        return model, r2

    def train_XGB(self, features, targets, model_name, test_size = 0.25):
        pass

    def train_svr(self, features, targets, model_name, test_size = 0.25, kernel = "rbf"):
        """Trains a SVR model.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVR
        from sklearn.externals import joblib
        
        # Splitting dataset and building the model
        x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size = test_size, random_state = 42)
        model = SVR(kernel = kernel)
        model.fit(x_train, y_train)

        # Saving the model
        model_name = model_name
        joblib.dump(model, model_name) 

        # Testing model and plotting results
        prediction = model.predict(x_test)
        print(prediction)
        r2 = r2_score(y_test, prediction)

        return model, r2

################################################################
########## Random Search with Cross-Validation
################################################################
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV 

    def rf_randomsearch(self, features, targets):
        # Splitting dataset and building the model
        x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size = test_size, 
                                                            random_state = 42)
        # Number of trees
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap} 

        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rf = RandomForestRegressor()
        model_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, 
                                       cv = 3, verbose=2, random_state=42, n_jobs = -1)
        model_random.fit(x_train, y_train)
        best_parameters = model_random.best_params 

        # Evaluate performance
        predictions = model.predict(x_test)
        errors = abs(predictions - y_test)
        mape = 100 * np.mean(errors / y_test)
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        
        return best_parameters, accuracy     

################################################################
########## Implementation
################################################################

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

    print(features)

    # if args["model"] == "Random Forest":
    #     model = Models()
    #     rf, r2 = model.train_randomforest(features, targets, model_name = "RF1.sav")
    #     print(r2)

    if args["model"] == "Random Forest":
        model = Models()
        best_parameters, accuracy = model.rf_randomsearch(features, targets)
        print(best_parameters)
        print(accuracy)

    if args["model"] == "SVR":
        model = Models()
        features = dataset.scaling_values(features)
        svr, r2 = model.train_svr(features, targets, model_name = "SVR1.sav")

    