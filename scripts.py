import platform

#from sklearn._typing import ArrayLike, Float, MatrixLike
from amplpy import AMPL
import subprocess
import numpy as np
import pandas as pd
from dataclasses import dataclass
from platform import system
from time import time
temp_fname = "temp.txt"
# get operating system

os_name = platform.system()

tol = 10e-3
class SVM_data:
    """
    Class to store the data of one instance ofthe SVM problem
    Train contains the training data
    Test contains the test data (with misclassified points)
    cTest contains the test data (with misclassified points corrected)
    """

    m: int
    nu: float
    test_frac: float
    seed: int
    X_train: pd.DataFrame
    Y_train: pd.DataFrame
    cY_train: pd.DataFrame
    X_test: pd.DataFrame
    Y_test: pd.DataFrame
    cY_test: pd.DataFrame

    def __init__(self, m: float, test_frac: float = 0.2, seed: int = 42):
        self.m = m
        self.test_frac = test_frac
        self.seed = seed
        (self.X_train, self.Y_train, self.cY_train), (
            self.X_test,
            self.Y_test,
            self.cY_test,
        ) = generate_dataset(m, test_frac, seed)
    def get_data(self):
        return self.X_train, self.Y_train, self.X_test,self.Y_test
    

@dataclass
class solution:
    """
    Class to store the solution of the SVM problem. Atributes can be empty
    """

    gamma: float
    lambda_: float
    w: float
    s: float

    def __init__(self, lambda_, gamma, w, s):
        self.gamma = gamma
        self.w = w
        self.s = s
        self.lambda_ = lambda_


def generate_dataset(m: float, test_frac: float, seed: int = 42, n: int = 4):
    """
    Creates a dataset of size m with test_frac of test data using the seed and returns a tuple of pandas dataframes X,Y
    divided in train, test and test corrected.
    The total number of observations is m + test_frac*m
    Parameters:
    ----------
    m: int 
        Number of train observations to generate
    test_frac: float
        Fraction of test observations to generate
    seed: int
        Seed for the random number generator
    """
    x = m + test_frac * m # total number of observations
    # Different command for different OS, random seed will not generate the same data
    # in different OS
    if system() == "Linux":
        subprocess.run(["./gensvmdat", temp_fname, str(x), str(seed)])
    elif system() == "Darwin":
        subprocess.run(["./gensvmdat-mac", temp_fname, str(x), str(seed)])

    # read data and split in train and test. The asterisk is removed from the labels
    # a corrected version of the test data is also created
    df = pd.read_csv(temp_fname, sep=" ", header=None)
    X_train = df.iloc[:m, :n]
    X_test = df.iloc[m:, :n]
    Y_train = df.iloc[:m, -1].str.replace("*", "", regex=False)
    Y_train = Y_train.astype(float)
    Y_test = df.iloc[m:, -1].str.replace("*", "", regex=False)
    Y_test = Y_test.astype(float)
    cY_train = df.iloc[:m, -
                       1].replace({"1.0*": "-1", "-1.0*": "1"}).astype(float)
    cY_test = df.iloc[m:, -
                      1].replace({"1.0*": "-1", "-1.0*": "1"}).astype(float)
    
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    Y_train = Y_train.to_numpy()
    Y_test = Y_test.to_numpy()
    cY_train = cY_train.to_numpy()
    cY_test = cY_test.to_numpy()

    return (
        (X_train, Y_train, cY_train),
        (X_test, Y_test, cY_test),
    )




def primal_model(ampl):
    """
    Creates the primal model in AMPL
    """
    model_primal = r"""
    reset;
    # PARAMETERS
    param m; 	# number of rows
    param n;    # number of columns
    param nu;   # nu parameter

    param X {1..m,1..n}; # data matrix
    param Y {1..m};      # labels

    # VARIABLES
    var w{1..n}; 
    var gamma;	
    var s{1..m};

    # OBJECTIVE function
    minimize fx:
            1/2*sum{i in 1..n}w[i]**2 + nu*sum{i in 1..m}s[i];
    # CONSTRAINTS
    subject to gx1 {i in 1..m}:
            Y[i]*((sum{j in 1..n}X[i,j]*w[j])+gamma)+s[i] >= 1;

    subject to gx2 {i in 1..m}:
            s[i] >= 0;
    """
    ampl.eval(model_primal)


def dual_model(ampl, kernel="linear"):
    """
    Creates the dual model in AMPL
    """
    if kernel == "linear":
        model_dual = r"""
        reset;
        # PARAMETERS
        param m; 	# number of rows
        param n;    # number of columns
        param nu;   # nu parameter

        param X {1..m,1..n}; # data matrix
        param Y {1..m};      # labels

        # VARIABLES
        var lambda{1..m} >= 0, <= nu;

        # OBJECTIVE function
        maximize q:
                sum{i in 1..m}lambda[i]
                -1/2*sum{i in 1..m, j in 1..m}lambda[i]*Y[i]*lambda[j]*Y[j]*(sum{k in 1..n}X[i,k]*X[j,k]); 

        # CONSTRAINTS                    
        subject to hx:
                sum{i in 1..m}lambda[i]*Y[i] = 0;
        """
    elif kernel == "gaussian":
            # return exp(-1/(2*sigma**2)*(sum{k in 1..n}(X[i,k]-X[j,k])**2));
        model_dual = r"""
        reset;
        # PARAMETERS
        param m; 	# number of rows
        param n;	# number of columns
        param nu;   # nu parameter
        param sigma;    # sigma parameter

        param X {1..m,1..n}; # data matrix
        param Y {1..m};     # labels
        param K {1..m,1..m}; # kernel matrix

        # VARIABLES
        var lambda{1..m} >= 0, <= nu;

        # OBJECTIVE function
        maximize q:
                sum{i in 1..m}lambda[i]
                -1/2*sum{i in 1..m, j in 1..m}lambda[i]*Y[i]*lambda[j]*Y[j]*K[i,j]; 

        # CONSTRAINTS
        subject to hx:
                sum{i in 1..m}(lambda[i]*Y[i]) = 0;
        
        """
    ampl.eval(model_dual)


def set_parameters(X, Y, m, n,nu, ampl,sigma = 1, kernel="linear"):
    """
    Sets the parameters of the model in AMPL with the data. If kernel is gaussian
    the sigma parameter is also set
    """
    ampl.param["m"] = m 
    ampl.param["n"] = n
    ampl.param["nu"] = nu

    if kernel == "gaussian":
        ampl.param["sigma"] = sigma
        K = np.zeros((m, m))   
        for i, j in np.ndindex(K.shape):
            K[i, j] = kernel_func(X[i, :], X[j, :], sigma=sigma)
            ampl.param["K"][i + 1, j + 1] = K[i, j]
    
    # We pass the data to AMPL
    for i in range(m):
        for j in range(n):
            ampl.param["X"][i + 1, j + 1] = X[i, j]
    for i in range(m):
        ampl.param["Y"][i + 1] = Y[i]


def solve(ampl: AMPL, solver: str = "ipopt", problem: str = "primal"):
    """
    Solves the model in AMPL with the solver specified and returns the solution
    in a solution object.
    Different solves migth produce numeric precision differences in the solution and
    performance might vary.
    The problem can be primal or dual.

    If the problem is primal, the solution contains gamma, w and s
    If the problem is dual, the solution contains lambda and the rest of the variables are None
    """
    print(solver)
    ampl.option["solver"] = solver
    solve_output = ampl.get_output("solve;")
    if ampl.get_value("solve_result")=="solved":
        if problem == "primal":
            w = ampl.get_variable("w").get_values().to_pandas().to_numpy()
            gamma = ampl.get_variable("gamma").get_values().to_pandas().to_numpy()
            s = ampl.get_variable("s").get_values().to_pandas().to_numpy()
            return solution(gamma=gamma, w=w, s=s, lambda_=None),solve_output
        elif problem == "dual":
            lambda_ = ampl.get_variable(
                "lambda").get_values().to_pandas().to_numpy()
            return solution(lambda_=lambda_, gamma=None, w=None, s=None),solve_output
    else:
        print("Not solved")
        return None, solve_output

def get_accuracy_linear(X_test, Y_test, w, gamma):
    """
    Calculates the accuracy of the model given w and gamma
    """
    test_size = len(X_test)
    y_pred = predict_linear(X_test,  w, gamma)
    misclassifications = 0

    for i in range(test_size):
        if y_pred[i] != Y_test[i]:
            misclassifications += 1

    accuracy = (test_size - misclassifications) / test_size

    return accuracy

def kernel_func(xi, xj, sigma=1):
    """
    Calculates the gaussian kernel function    
    """
    return np.exp(-1/(2*sigma**2)*np.linalg.norm(xi-xj)**2)


def predict_linear(X_test, w, gamma):
    """
    Predicts the labels of the test data given w and gamma if the kernel is linear
    and therefore the parameters w and gamma are available.

    Parameters:
    -----------
    X_test: np.array
        Array of test data
    w: np.array
        w parameter calculated by the model
    gamma: float
        gamma parameter calculated by the model
    
    Returns:
    --------
    y_pred: np.array
        Array of predicted labels
    """
    test_size = len(X_test)
    y_pred = np.zeros(test_size)
    for i in range(test_size):
        y_pred[i] = np.sign(gamma+sum(w[j]*X_test[i,j] for j in range(len(w))))

    return y_pred




def predict_kernel(X_test, X_train, Y_train, lambda_, gamma, sigma=1):
    """
    Predicts the labels of the test data given lambda and gamma if the kernel is gaussian
    
    Parameters:
    -----------
    X_test: np.array
        Array of test data
    X_train: np.array
        Array of train data used to train the model
    Y_train: np.array
        Array of train labels used to train the model
    lambda_: np.array
        Array of lambda calculated by the model
    gamma: float
        gamma parameter calculated by the model
    sigma: float
        sigma parameter of the gaussian kernel
    
    Returns:
    --------
    y_pred: np.array
        Array of predicted labels
    """   

    test_size = len(X_test)
    y_pred = np.zeros(test_size)
    for j in range(test_size):
        w_phi = sum(lambda_[i]*Y_train[i]*kernel_func(X_test[j,:], X_train[i,:], sigma=sigma)	
                    for i in range(len(lambda_)))
        y_pred[j] = np.sign(gamma + w_phi)
    return y_pred

def get_model_param(X, Y, lambda_,  nu, 
                    kernel="linear", sigma=1):
    """
    Calculates w and gamma from lambda
    If kernel is gaussian only calculates gamma
    """
    m, n = X.shape
    w = np.zeros(n)
    if kernel == "linear":
        for j in range(n):
            w[j] = np.sum(lambda_[i] * Y[i] * X[i, j]
                          for i in range(m))

        # Gamma is calculated using a support vector
        gamma = None
        for i in range(m):
            if 0.01 < lambda_[i] < nu * 0.99:
                # A support vector point was found
                gamma = 1 / Y[i] - np.sum(w[j] * X[i, j] for j in range(n))
                break
    elif kernel == "gaussian":
        k = 0
        for i in range(m):
            if tol < lambda_[i] < nu:
                k = i
                break
        w_phi = sum(lambda_[i]*Y[i]*kernel_func(X[i,:], X[k,:],sigma) for i in range(len(lambda_)))
        gamma = 1/Y[k] - w_phi

    return w, gamma

import re

def get_metrics(output:str):
    """
    From the ampl output, returns the metrics of the solve
    """

    metrics = {}
    iterations = re.search(r'Number of Iterations\.*: +(\d+)', output)
    ipopt_cpu_secs = re.search(r'Total CPU secs in IPOPT \(w/o function evaluations\)   = +([\d.]+)', output)
    nlp_cpu_secs = re.search(r'Total CPU secs in NLP function evaluations           = +([\d.]+)', output)
    if iterations:
        metrics["num_iterations"] = int(iterations.group(1))
    if ipopt_cpu_secs:
        metrics['time_ipopt'] = float(ipopt_cpu_secs.group(1))
    if nlp_cpu_secs:
        metrics['time_npl'] = float(nlp_cpu_secs.group(1))
        return metrics


from sklearn.base import BaseEstimator, ClassifierMixin
class  SVM_ampl(BaseEstimator, ClassifierMixin):
    """
    Class to create a SVM model using AMPL. The model can be primal or dual and the kernel can be linear or gaussian.
    The parameters of the model are calculated using AMPL and the solution is stored in the object.

    The use of sklearn is to be able to use the model with the sklearn cross validation functions and to be able to
    compare the results with the sklearn SVM model.

    Parameters:
    ----------
    nu: float
        nu parameter of the SVM model
    mode: str
        mode of the model, can be "primal" or "dual"
    kernel: str
        kernel of the model, can be "linear" or "gaussian"
    sigma: float
        sigma parameter of the gaussian kernel  
    
    Attributes:
    -----------
    gamma: float
    w: np.array
    s: np.array 
    lambda_: np.array
    metrics: dict
        Dictionary with the metrics of the solve (number of iterations, 
        ampl solve time and nlp solve time (if ipopt is used))

    Methods:
    --------
    fit(X,y)
        Fits the model with the data X and labels y
    predict(X)
        Predicts the labels of the data X

            
    """
    def __init__(self, nu=1, n = 4, type = "primal", sigma=None, solver = "highs"):
        self.type = type
        if type == "primal":
            self.mode = "primal"
            self.kernel = "linear"
        elif type == "dual_linear":
            self.mode = "dual"
            self.kernel = "linear" 
        elif type == "dual_gaussian":
            self.mode = "dual"
            self.kernel = "gaussian"
        self.n = n
        self.nu = nu
        self.sigma = sigma
        self.solver= solver

    def fit(self, X, y):
        """
        Fits the model with the data X and labels y
        Parameters:
        ----------
        X: np.array
            Array of data
        y: np.array
            Array of labels

        Returns:
        --------
        self: SVM_ampl
            The fitted model
              """
        m = len(X)
        ampl = AMPL()
        
        self.X = X
        self.y = y
        if self.sigma == None:
            self.sigma = np.sqrt(X.shape[1]/2)
        # depending on the mode and kernel, the model is created

        if self.mode == "primal":
            ampl = AMPL()
            primal_model(ampl)
            set_parameters(X,y,m,self.n,self.nu,ampl)
            # measure solve time
            start = time()
            solution, self.output = solve(ampl,self.solver,self.mode)
            end = time()
            self.solve_time = end-start
            self.gamma = ampl.get_variable("gamma").get_values().to_pandas().to_numpy()
            self.w = ampl.get_variable("w").get_values().to_pandas().to_numpy()
            self.s = ampl.get_variable("s").get_values().to_pandas().to_numpy()
            self.obj = ampl.get_objective("fx").get().value()
            # self.const = (ampl.get_constraint("gx1").get().dual(),ampl.get_constraint("gx2").get().dual())
        elif self.mode == "dual" and self.kernel == "linear":
            ampl = AMPL()
            dual_model(ampl)
            set_parameters(X,y,m,self.n,self.nu,ampl)
            start = time()
            solution, self.output = solve(ampl,self.solver,self.mode)
            end = time()
            self.solve_time = end-start
            solution = solve(ampl,self.solver,self.mode)
            self.la = ampl.get_variable("lambda").get_values().to_pandas().to_numpy()
            self.w, self.gamma = get_model_param(X,y,self.la, self.nu,kernel="linear")
            self.obj = ampl.get_objective("q").get().value()
            # self.const = ampl.get_constraint("hx").get().dual()
        elif self.mode == "dual" and self.kernel == "gaussian":
            ampl = AMPL()
            dual_model(ampl, kernel="gaussian")
            set_parameters(X,y,m,self.n,self.nu,ampl,sigma=self.sigma,kernel="gaussian")
            start = time()
            solution, self.output = solve(ampl,self.solver,self.mode)
            end = time()
            self.solve_time = end-start
            self.lambda_ = ampl.get_variable("lambda").get_values().to_pandas().to_numpy()
            self.w, self.gamma = get_model_param(X,y, self.lambda_,self.nu,kernel="gaussian",sigma=self.sigma)
            self.obj = ampl.get_objective("q").get().value()
            # self.const = ampl.get_constraint("hx").get().dual()
        try: 
            self.metrics = get_metrics(self.output)
        except Exception as e:
            self.metrics = None
            print("Could not get metrics")
            print(str(e))

        return self
    def predict(self, X):
        """
        Predicts the labels of the data X
        Returns:
        --------
        y_pred: np.array
            Array of predicted labels
        """
        if self.mode == "primal":
            return predict_linear(X,self.w,self.gamma)
        elif self.mode == "dual" and self.kernel == "linear":
            return predict_linear(X, self.w, self.gamma)
        elif self.mode == "dual" and self.kernel == "gaussian":
            print("predicting with gaussian kernel")
            return predict_kernel(X, self.X, self.y,self.lambda_, self.gamma,self.sigma)