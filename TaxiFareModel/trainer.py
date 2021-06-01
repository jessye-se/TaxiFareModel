# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return pipe
    
    def run(self):
        # implement train() function
        '''returns a trained pipelined model'''
        pipeline.fit(X_train, y_train)
        return pipeline

    def evaluate(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test 
        '''returns the value of the RMSE'''
        y_pred = pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse

if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
